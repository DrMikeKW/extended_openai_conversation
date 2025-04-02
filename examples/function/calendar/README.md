## Objective
<img width="300" alt="스크린샷 2023-10-31 오후 9 04 56" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/7a6c6925-a53e-4363-a93c-45f63951d41b">

## Function

### 1. get_events
```yaml
- spec:
    name: get_events
    description: Use this function to get a list of calendar events.
    parameters:
      type: object
      properties:
        target:
          type: object
          properties:
            entity_id:
              type: string
              description: The entity_id of the calendar. Either entity_id or label_id is required.
            label_id:
              type: string
              description: The label_id of calendars. Either entity_id or label_id is required.
          anyOf:
            - required: ["entity_id"]
            - required: ["label_id"]
        start_date_time:
          type: string
          description: The start date time in '%Y-%m-%dT%H:%M:%S%z' format.
        final_time:
          type: object
          properties:
            end_date_time:
              type: string
              description: The end date time in '%Y-%m-%dT%H:%M:%S%z' format.
            duration:
              type: object
              description: Return active events from start_date_time until the specified duration. Expressed as a dictionary with days, hours, or minutes.
          oneOf:
            - required: ["end_date_time"]
            - required: ["duration"]
      required:
        - start_date_time
        - final_time
        - target
  function:
    type: composite
    sequence:
      - type: script
        sequence:
          - variables:
              target_data: >
                {% set data = {} %}
                {% if target.entity_id is defined and has_value(target.entity_id) and 'entity_id' in target and target.entity_id.split('.')[0] == 'calendar' %}
                  {% set data = dict(data, **{'entity_id': target.entity_id}) %}
                {% endif %}
                {% if target.label_id|default('') in labels() %}
                  {% set data = dict(data, **{'label_id': target.label_id}) %}
                {% endif %}
                {{ data | tojson }}
              event_data: >
                {% set data = {} %}
                {% set data = dict(data, **{'start_date_time': start_date_time}) %}
                {% if 'end_date_time' in final_time and final_time.end_date_time %}
                  {% set data = dict(data, **{'end_date_time': (final_time.end_date_time | default(none))}) %}
                {% elif 'duration' in final_time and final_time.duration %}
                  {% set data = dict(data, **{'duration': (final_time.duration | default(none))}) %}
                {% endif %}
                {{ data | tojson }}
              data_filled: >
                {% set target_dict = (target_data | from_json) if target_data is string else target_data %}
                {% set event_dict = (event_data | from_json) if event_data is string else event_data %}
                {{ target_dict | length > 0 and event_dict | length > 0 }}
          - condition: template
            value_template: "{{ data_filled }}"
          - service: calendar.get_events
            data: "{{ event_data }}"
            target: "{{ target_data }}"
            response_variable: _function_result
```



### 1(A). Previous version get_events
```yaml
- spec:
    name: get_events
    description: Use this function to get list of calendar events.
    parameters:
      type: object
      properties:
        start_date_time:
          type: string
          description: The start date time in '%Y-%m-%dT%H:%M:%S%z' format
        end_date_time:
          type: string
          description: The end date time in '%Y-%m-%dT%H:%M:%S%z' format
      required:
      - start_date_time
      - end_date_time
  function:
    type: script
    sequence:
    - service: calendar.get_events
      data:
        start_date_time: "{{start_date_time}}"
        end_date_time: "{{end_date_time}}"
      target:
        entity_id:
        - calendar.[YourCalendarHere]
        - calendar.[MoreCalendarsArePossible]
      response_variable: _function_result
```

### 2. create_event
```yaml
- spec:
    name: create_event
    description: Adds a new calendar event.
    parameters:
      type: object
      properties:
        summary:
          type: string
          description: Defines the short summary or subject for the event.
        description:
          type: string
          description: A more complete description of the event than the one provided by the summary.
        start_date_time:
          type: string
          description: The date and time the event should start.
        end_date_time:
          type: string
          description: The date and time the event should end.
        location:
          type: string
          description: The location
      required:
      - summary
  function:
    type: script
    sequence:
      - service: calendar.create_event
        data:
          summary: "{{summary}}"
          description: "{{description}}"
          start_date_time: "{{start_date_time}}"
          end_date_time: "{{end_date_time}}"
          location: "{{location}}"
        target:
          entity_id: calendar.[YourCalendarHere]
```
