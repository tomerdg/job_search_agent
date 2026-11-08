{# Use the folder schema as is (staging, intermediate, marts) instead of "main_staging". #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name | trim if custom_schema_name else target.schema }}
{%- endmacro %}
