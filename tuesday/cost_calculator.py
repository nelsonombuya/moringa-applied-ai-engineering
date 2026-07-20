def estimate_monthly_cost(
    avg_input_tokens,
    avg_output_tokens,
    daily_messages,
    days_per_month=30,
):
    """
    Calculate estimated monthly API costs for AfyaPlus.
    GPT-4o-mini pricing:
    - Input:  $0.15 per 1,000,000 tokens
    - Output: $0.60 per 1,000,000 tokens
    """
    INPUT_PRICE_PER_M = 0.15
    OUTPUT_PRICE_PER_M = 0.60
    monthly_messages = daily_messages * days_per_month
    total_input_tokens = monthly_messages * avg_input_tokens
    total_output_tokens = monthly_messages * avg_output_tokens
    input_cost = (total_input_tokens / 1_000_000) * INPUT_PRICE_PER_M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
    total_cost = input_cost + output_cost
    return {
        "monthly_messages": monthly_messages,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_monthly": total_cost,
        "cost_per_message": total_cost / monthly_messages,
    }


# Test: 5,000 messages/day, avg 100 input + 150 output tokens
result = estimate_monthly_cost(
    avg_output_tokens=150,
    avg_input_tokens=100,
    daily_messages=5000,
)
print(f"Monthly cost estimate: ${result['total_monthly']:.2f}")
print(f"Monthly messages: {result['monthly_messages']:,}")
print(f"Input cost:  ${result['input_cost']:.2f}")
print(f"Output cost: ${result['output_cost']:.2f}")
print(f"Total monthly: ${result['total_monthly']:.2f}")
print(f"Cost per message: ${result['cost_per_message']:.6f}")
