
"""OpenAI Conversation integration with conversation history and GPT-5 support."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Literal

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import AuthenticationError, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice
import yaml

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_NAME, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError, TemplateError
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    intent,
    template,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_FUNCTIONS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_TOOLS,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
)
from .exceptions import (
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .helpers import get_function_executor, is_azure, validate_authentication
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

DATA_AGENT = "agent"

def _is_new_series_model(model: str) -> bool:
    """Return True for models that require `max_completion_tokens` and reject classic sampling knobs.
    Covers GPT-5 and o-series (o1/o3/o4). Safe no-op for other names.
    """
    try:
        result = model.startswith(("gpt-5", "o1", "o3", "o4"))
        _LOGGER.debug("model_check name=%s new_series=%s", model, result)
        return result
    except Exception as e:
        _LOGGER.debug("model_check error name=%s err=%s", model, e)
        return False

def _normalize_params_for_model(model: str, params: dict) -> dict:
    """Clone and normalize request params for the target model.
    - For new-series models: move `max_tokens` -> `max_completion_tokens` and drop sampling/penalty fields.
    - For legacy models: leave as-is.
    """
    before_keys = list(params.keys())
    p = dict(params)
    if _is_new_series_model(model):
        if "max_tokens" in p:
            p["max_completion_tokens"] = p.pop("max_tokens")
        # Drop fields rejected by GPT-5/o-series
        for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            p.pop(k, None)
    after_keys = list(p.keys())
    _LOGGER.debug(
        "param_norm model=%s keys_before=%s keys_after=%s has_mct=%s",
        model,
        before_keys,
        after_keys,
        "max_completion_tokens" in p,
    )
    return p



async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""
    _LOGGER.info("setup start")
    await async_setup_services(hass, config)
    _LOGGER.info("setup done")
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""
    _LOGGER.info("setup_entry start entry_id=%s base_url=%s", entry.entry_id, entry.data.get(CONF_BASE_URL))
    try:
        await validate_authentication(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
            skip_authentication=entry.data.get(CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION),
        )
    except AuthenticationError as err:
        _LOGGER.error("auth invalid_api_key err=%s", err)
        return False
    except OpenAIError as err:
        _LOGGER.error("auth openai_error err=%s", err)
        raise ConfigEntryNotReady(err) from err

    agent = OpenAIAgent(hass, entry)
    _LOGGER.info("setup_entry agent_created for entry_id=%s", entry.entry_id)

    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[CONF_API_KEY] = entry.data[CONF_API_KEY]
    data[DATA_AGENT] = agent

    conversation.async_set_agent(hass, entry, agent)
    _LOGGER.info("setup_entry complete entry_id=%s", entry.entry_id)
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI Conversation."""
    _LOGGER.info("unload_entry start entry_id=%s", entry.entry_id)
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    _LOGGER.info("unload_entry complete entry_id=%s", entry.entry_id)
    return True

# Custom ChatLog class to manage conversation history, in the future replace with the official home assistant conversation agent history
class ChatLog:
    """Minimal class to manage the conversation log."""
    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.content: list[dict] = []

class OpenAIAgent(conversation.AbstractConversationAgent):
    """OpenAI conversation agent with integrated conversation history support."""
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        # Initialize the dictionary for chat logs if not already present
        self.hass.data.setdefault("extended_openai_chat_logs", {})
        base_url = entry.data.get(CONF_BASE_URL)
        _LOGGER.info("agent_init base_url=%s azure=%s", base_url, is_azure(base_url))
        if is_azure(base_url):
            self.client = AsyncAzureOpenAI(
                api_key=entry.data[CONF_API_KEY],
                azure_endpoint=base_url,
                api_version=entry.data.get(CONF_API_VERSION),
                organization=entry.data.get(CONF_ORGANIZATION),
                http_client=get_async_client(hass),
            )
        else:
            self.client = AsyncOpenAI(
                api_key=entry.data[CONF_API_KEY],
                base_url=base_url,
                organization=entry.data.get(CONF_ORGANIZATION),
                http_client=get_async_client(hass),
            )
        _LOGGER.info("agent_init client_ready entry_id=%s", entry.entry_id)

    def _get_chat_log(self, conversation_id: str) -> ChatLog:
        """Retrieve or create the chat log for the specified conversation_id."""
        logs = self.hass.data["extended_openai_chat_logs"]
        if conversation_id not in logs:
            _LOGGER.debug("chat_log create conversation_id=%s", conversation_id)
            logs[conversation_id] = ChatLog(conversation_id)
        else:
            _LOGGER.debug("chat_log reuse conversation_id=%s", conversation_id)
        return logs[conversation_id]

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: ChatLog,
    ) -> conversation.ConversationResult:
        _LOGGER.debug("handle_message start conv_id=%s text_len=%s", user_input.conversation_id, len(user_input.text or ""))
        exposed_entities = self.get_exposed_entities()

        # If the log is empty, insert the system message (with the prompt)
        if not chat_log.content:
            try:
                system_message = self._generate_system_message(exposed_entities, user_input)
            except TemplateError as err:
                _LOGGER.error("prompt_render error err=%s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=chat_log.conversation_id
                )
            chat_log.content.append(system_message)
            _LOGGER.debug("handle_message added_system_message conv_id=%s", chat_log.conversation_id)
        
        # Optimize context for GPT-5 to reduce lag
        if _is_new_series_model(self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)):
            # For GPT-5, limit conversation history to last 5 exchanges to reduce processing time
            max_history = 10  # 5 exchanges (user + assistant pairs)
            if len(chat_log.content) > max_history:
                # Keep system message and last few exchanges
                chat_log.content = [chat_log.content[0]] + chat_log.content[-max_history+1:]
                _LOGGER.debug("GPT-5 context optimization: reduced history from %d to %d messages", 
                             len(chat_log.content) + max_history - 1, len(chat_log.content))

        # Append the user's message
        user_message = {"role": "user", "content": user_input.text}
        if self.entry.options.get(CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME):
            user = user_input.context.user_id
            if user is not None:
                user_message[ATTR_NAME] = user
        chat_log.content.append(user_message)
        _LOGGER.debug("handle_message appended_user_message conv_id=%s total_messages=%d", chat_log.conversation_id, len(chat_log.content))
        try:
            query_response = await self.query(user_input, chat_log.content, exposed_entities, 0)
        except OpenAIError as err:
            _LOGGER.error("openai_error err=%s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to OpenAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=chat_log.conversation_id, continue_conversation=self.continue_conversation(chat_log.content)
            )
        except HomeAssistantError as err:
            _LOGGER.error("ha_error err=%s", err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=chat_log.conversation_id
            )
         # Append the assistant's response to the conversation history
        chat_log.content.append(query_response.message.model_dump(exclude_none=True))

         # Optionally update the log in hass.data
        self.hass.data["extended_openai_chat_logs"][chat_log.conversation_id] = chat_log
        _LOGGER.debug("handle_message appended_assistant_message conv_id=%s total_messages=%d", chat_log.conversation_id, len(chat_log.content))
        self.hass.bus.async_fire(
            EVENT_CONVERSATION_FINISHED,
            {
                "response": query_response.response.model_dump(),
                "user_input": user_input,
                "messages": chat_log.content,
            },
        )
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(query_response.message.content)
        _LOGGER.debug("handle_message end conv_id=%s", chat_log.conversation_id)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=chat_log.conversation_id, continue_conversation=self.continue_conversation(chat_log.content)
        )
    def continue_conversation(self, content) -> bool:
        """Return whether the conversation should continue."""
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("chat_log.content: %s", json.dumps(content)[:1000])
        if not content:
            return False
        last_msg = content[-1]
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("last message: %s", json.dumps(last_msg)[:1000])
        return (
            last_msg["role"] == "assistant"
            and last_msg.get("content") is not None
            and last_msg.get("content", "").strip().endswith(("?", ";"))
        )

    def _generate_system_message(
        self, exposed_entities, user_input: conversation.ConversationInput
    ) -> dict:
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        
        # The original prompt already contains tool usage instructions
        # No need to add more - the existing prompt is comprehensive
        enhanced_prompt = raw_prompt
        
        # Use the template to generate the final prompt, similar to the base integration
        # For GPT-5 models, reduce token usage by selecting only relevant entities
        entities_for_prompt = exposed_entities
        if _is_new_series_model(self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)):
            try:
                selected = self._select_relevant_entities(exposed_entities, user_input.text or "")
                if selected:
                    entities_for_prompt = selected
                    _LOGGER.debug(
                        "entity_filter applied model=gpt-5 before=%d after=%d",
                        len(exposed_entities),
                        len(entities_for_prompt),
                    )
            except Exception as e:
                _LOGGER.debug("entity_filter error err=%s", e)

        prompt = template.Template(enhanced_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": entities_for_prompt,
                "current_device_id": user_input.device_id,
            },
            parse_result=False,
        )
        _LOGGER.debug("system_message length=%d", len(prompt or ""))
        return {"role": "system", "content": prompt}

    def _should_use_tools(self, messages: list, user_text: str, model: str) -> bool:
        """Determine if the current query needs function calls based on content analysis.
        
        This prevents GPT-5 from making unnecessary function calls for simple queries
        like math problems or basic questions that don't require device control.
        """
        # Simple queries that don't need tools
        simple_patterns = [
            r"what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+",  # Math: "what is 6+4"
            r"what\s+time\s+is\s+it",               # Time questions
            r"what\s+day\s+is\s+it",                # Date questions
            r"how\s+are\s+you",                     # Greetings
            r"hello",                                # Greetings
            r"hi\b",                                 # Greetings
            r"good\s+(morning|afternoon|evening)",   # Greetings
        ]
        
        # Check if user query matches simple patterns
        user_text_lower = user_text.lower().strip()
        for pattern in simple_patterns:
            if re.search(pattern, user_text_lower):
                _LOGGER.debug("simple_query_detected pattern=%s text=%s", pattern, user_text)
                return False
        
        # Check if this is a device control request
        control_keywords = [
            "turn on", "turn off", "switch", "control", "set", "adjust", "dim", "brighten",
            "open", "close", "lock", "unlock", "start", "stop", "play", "pause"
        ]
        
        for keyword in control_keywords:
            if keyword in user_text_lower:
                _LOGGER.debug("control_query_detected keyword=%s text=%s", keyword, user_text)
                return True
        
        # For GPT-5, be more conservative about tool usage
        if _is_new_series_model(model):
            # Only use tools if explicitly requested or for device control
            if any(keyword in user_text_lower for keyword in control_keywords):
                return True
            
            # For status queries about lights/devices, don't use tools if the info is in the system prompt
            status_keywords = ["what", "which", "how many", "are", "is"]
            if any(keyword in user_text_lower for keyword in status_keywords):
                # Check if this is asking about device states (lights, switches, etc.)
                device_queries = ["light", "switch", "sensor", "fan", "door", "window", "temperature", "humidity"]
                if any(device in user_text_lower for device in device_queries):
                    _LOGGER.debug("status_query_detected skipping_tools text=%s", user_text)
                    return False
            
            return False
        
        # For legacy models, use existing behavior
        return True

    def _select_relevant_entities(self, exposed_entities: list, user_text: str) -> list:
        """Select a relevant subset of entities based on the user's query.

        - Filters by domain if the query mentions a domain (e.g., lights, switches, fans)
        - Further filters by keywords (e.g., office, kitchen) against name/aliases
        - Falls back gracefully if filtering would return nothing
        - Caps the result to a reasonable size to keep tokens low for GPT-5
        """
        if not user_text:
            return []

        text_lower = user_text.lower()
        # Domain keywords mapping
        domain_keywords = {
            "light": {"light", "lights"},
            "switch": {"switch", "switches"},
            "fan": {"fan", "fans"},
            "sensor": {"sensor", "sensors", "temperature", "humidity", "aqi", "uv"},
            "media_player": {"media", "tv", "roku", "sonos", "speaker", "speakers"},
            "cover": {"cover", "door", "gate", "garage"},
            "scene": {"scene", "scenes", "mode", "modes"},
            "weather": {"weather"},
            "binary_sensor": {"motion", "door", "window", "contact", "occupancy"},
            "script": {"script", "automation"},
        }

        target_domains = set()
        for domain, keywords in domain_keywords.items():
            if any(k in text_lower for k in keywords):
                target_domains.add(domain)

        # Initial filter by domain when mentioned
        filtered = exposed_entities
        if target_domains:
            filtered = [
                e for e in exposed_entities
                if isinstance(e.get("entity_id"), str) and e["entity_id"].split(".")[0] in target_domains
            ]

        # Tokenize text and filter by name/aliases containing any keyword (e.g., area like "office")
        tokens = set(re.findall(r"[a-z0-9_]+", text_lower))
        def matches_keywords(entity: dict) -> bool:
            name = (entity.get("name") or "").lower()
            aliases = [a.lower() for a in (entity.get("aliases") or [])]
            hay = " ".join([name] + aliases)
            return any(tok in hay for tok in tokens if len(tok) >= 3)

        keyword_matches = [e for e in filtered if matches_keywords(e)]
        if keyword_matches:
            filtered = keyword_matches

        # If everything got filtered out, return empty to signal no change
        if not filtered:
            return []

        # Cap to a reasonable number for GPT-5 performance without losing correctness
        MAX_ENTITIES = 200
        if len(filtered) > MAX_ENTITIES:
            filtered = filtered[:MAX_ENTITIES]

        _LOGGER.debug(
            "entity_filter domains=%s tokens=%d result=%d",
            sorted(list(target_domains)) if target_domains else [],
            len(tokens),
            len(filtered),
        )
        return filtered

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:

        """Delegate to async_handle_message using the integrated chat_log."""
        chat_log = self._get_chat_log(user_input.conversation_id)
        _LOGGER.debug("process conv_id=%s", user_input.conversation_id)
        return await self.async_handle_message(user_input, chat_log)

    def get_exposed_entities(self):
        """Retrieve the exposed entities from Home Assistant."""
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)
            aliases = entity.aliases if entity and entity.aliases else []
            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": state.state,
                    "aliases": aliases,
                }
            )
        _LOGGER.debug("exposed_entities count=%d", len(exposed_entities))
        return exposed_entities
    def get_functions(self):
        """Load and prepare function definitions from configuration."""
        try:
            function = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(setting["function"]["type"])
                    setting["function"] = function_executor.to_arguments(setting["function"])
            _LOGGER.debug("functions loaded count=%d", len(result or []))
            return result
        except (InvalidFunction, FunctionNotFound) as e:
            _LOGGER.error("functions load error err=%s", e)
            raise e
        except Exception as e:
            _LOGGER.error("functions unexpected error err=%s", e)
            raise FunctionLoadFailed()

    async def truncate_message_history(self, messages: list[dict], exposed_entities, user_input: conversation.ConversationInput):
        """Truncate the conversation history and ensure the system message is always present."""

        strategy = self.entry.options.get(CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY)
        before_len = len(messages)
        if strategy == "clear":
            last_user_index = None
            for i in reversed(range(len(messages))):
                if messages[i]["role"] == "user":
                    last_user_index = i
                    break
            if last_user_index is not None:
                messages = [messages[0]] + messages[last_user_index:]
        if not messages or messages[0].get("role") != "system":
            system_message = self._generate_system_message(exposed_entities, user_input)
            messages.insert(0, system_message)
        after_len = len(messages)
        _LOGGER.debug("truncate_history strategy=%s before=%d after=%d", strategy, before_len, after_len)
        return messages

    async def query(
        self,
        user_input: conversation.ConversationInput,
        messages,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        """Send the conversation messages to OpenAI and handle the response."""
        model = self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        use_tools = self.entry.options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS)
        context_threshold = self.entry.options.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD)
        cfg_funcs = self.get_functions()
        functions = [s["spec"] for s in cfg_funcs] if cfg_funcs else []
        function_call = "auto"
        # Determine if this query needs function calls based on content and model
        needs_tools = self._should_use_tools(messages, user_input.text, model)
        
        if n_requests == self.entry.options.get(CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION):
            function_call = "none"
            needs_tools = False
        
        # Configure tool usage based on whether tools are actually needed
        if use_tools and needs_tools and len(functions) > 0:
            tool_kwargs = {
                "tools": [{"type": "function", "function": func} for func in functions],
                "tool_choice": function_call,
            }
        else:
            # Disable tools for simple queries or when not needed
            tool_kwargs = {}
            _LOGGER.debug("tools_disabled reason=%s model=%s", 
                         "not_needed" if not needs_tools else "max_calls_reached" if function_call == "none" else "no_functions", 
                         model)
        if len(functions) == 0:
            tool_kwargs = {}
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("prompt model=%s n_messages=%d use_tools=%s funcs=%d fcall=%s", model, len(messages), use_tools, len(functions), function_call)

        # >>> BAZEL PATCH START: normalized chat.completions.create payload
        # Start timing for performance monitoring
        start_time = asyncio.get_event_loop().time()
        
        # Build API request parameters with all required fields
        params = {
            "model": model,
            "messages": messages,
            "user": user_input.conversation_id,
            **tool_kwargs,
        }
        
        # Add legacy parameters that will be normalized for new-series models below
        params["max_tokens"] = max_tokens
        params["top_p"] = top_p
        params["temperature"] = temperature
        
        # Normalize parameters for GPT-5 and other new-series models
        params = _normalize_params_for_model(model, params)
        
        # Log detailed request information for debugging
        _LOGGER.debug(
            "request model=%s has_mct=%s max_tokens=%s temperature_included=%s",
            model,
            "max_completion_tokens" in params,
            params.get("max_tokens"),
            "temperature" in params,
        )
        
        # Set timeout based on model type to prevent excessive lag
        # GPT-5 and new-series models get 30s, legacy models get 15s
        timeout = 30 if _is_new_series_model(model) else 15
        
        # Log the complete prompt and parameters for debugging and monitoring
        _LOGGER.info("Prompt for %s: %s", model, json.dumps(messages))
        _LOGGER.info("Requesting OpenAI API with %ds timeout for model %s", timeout, model)
        
        # Execute API request with timeout protection
        try:
            response: ChatCompletion = await asyncio.wait_for(
                self.client.chat.completions.create(**params),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            _LOGGER.error("OpenAI API request timed out after %ds for model %s", timeout, model)
            raise HomeAssistantError(f"OpenAI API request timed out after {timeout} seconds")
        
        # Calculate and log performance metrics for monitoring
        end_time = asyncio.get_event_loop().time()
        response_time = end_time - start_time
        _LOGGER.info("OpenAI API response time: %.2fs for model %s", response_time, model)
        
        # Log the complete response for debugging and token analysis
        _LOGGER.info("Response %s", json.dumps(response.model_dump(exclude_none=True)))
        
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "response usage_total=%s finish_reason=%s choices=%d",
                getattr(response.usage, "total_tokens", None),
                getattr(response.choices[0], "finish_reason", None),
                len(response.choices or []),
            )
        if response.usage.total_tokens > context_threshold:
            messages[:] = await self.truncate_message_history(messages, exposed_entities, user_input)
        choice: Choice = response.choices[0]
        message = choice.message
        if choice.finish_reason == "function_call":
            _LOGGER.debug("function_call name=%s", getattr(message.function_call, "name", None))
            return await self.execute_function_call(user_input, messages, message, exposed_entities, n_requests + 1)
        if choice.finish_reason == "tool_calls":
            _LOGGER.debug("tool_calls count=%d", len(message.tool_calls or []))
            return await self.execute_tool_calls(user_input, messages, message, exposed_entities, n_requests + 1)
        if choice.finish_reason == "length":
            _LOGGER.debug("finish_reason length tokens=%s", getattr(response.usage, "completion_tokens", None))
            raise TokenLengthExceededError(response.usage.completion_tokens)
        _LOGGER.debug("query done conv_id=%s", user_input.conversation_id)
        return OpenAIQueryResponse(response=response, message=message)

    async def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        function_name = message.function_call.name
        _LOGGER.debug("execute_function_call name=%s", function_name)
        function = next(
            (s for s in (self.get_functions() or []) if s["spec"]["name"] == function_name),
            None,
        )
        if function is not None:
            return await self.execute_function(user_input, messages, message, exposed_entities, n_requests, function)
        _LOGGER.debug("execute_function_call missing name=%s", function_name)
        raise FunctionNotFound(function_name)

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
        function,
    ) -> OpenAIQueryResponse:
        function_executor = get_function_executor(function["function"]["type"])
        try:
            arguments = json.loads(message.function_call.arguments)
        except json.decoder.JSONDecodeError as err:
            _LOGGER.debug("execute_function bad_json name=%s err=%s", message.function_call.name, err)
            raise ParseArgumentsFailed(message.function_call.arguments) from err
        _LOGGER.debug("execute_function start name=%s", message.function_call.name)
        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )
        _LOGGER.debug("execute_function done name=%s", message.function_call.name)
        messages.append(
            {
                "role": "function",
                "name": message.function_call.name,
                "content": str(result),
            }
        )
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def execute_tool_calls(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        _LOGGER.debug("execute_tool_calls count=%d", len(message.tool_calls or []))

        """Execute tool calls if requested by OpenAI."""
        messages.append(message.model_dump(exclude_none=True))
        for tool in message.tool_calls:
            function_name = tool.function.name
            _LOGGER.debug("execute_tool name=%s id=%s", function_name, tool.id)
            function = next(
                (s for s in (self.get_functions() or []) if s["spec"]["name"] == function_name),
                None,
            )
            if function is not None:
                result = await self.execute_tool_function(user_input, tool, exposed_entities, function)
                messages.append(
                    {
                        "tool_call_id": tool.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(result),
                    }
                )
            else:
                _LOGGER.debug("execute_tool missing name=%s id=%s", function_name, tool.id)
                raise FunctionNotFound(function_name)
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def execute_tool_function(
        self,
        user_input: conversation.ConversationInput,
        tool,
        exposed_entities,
        function,
    ):
        """Execute a tool function."""
        function_executor = get_function_executor(function["function"]["type"])
        try:
            arguments = json.loads(tool.function.arguments)
        except json.decoder.JSONDecodeError as err:
            _LOGGER.debug("execute_tool_function bad_json name=%s err=%s", tool.function.name, err)
            raise ParseArgumentsFailed(tool.function.arguments) from err
        _LOGGER.debug("execute_tool_function start name=%s id=%s", tool.function.name, tool.id)
        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )
        _LOGGER.debug("execute_tool_function done name=%s id=%s", tool.function.name, tool.id)
        return result

class OpenAIQueryResponse:
    """Value object representing an OpenAI query response."""
    def __init__(self, response: ChatCompletion, message: ChatCompletionMessage) -> None:
        self.response = response
        self.message = message
