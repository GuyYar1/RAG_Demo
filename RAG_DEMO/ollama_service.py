import ollama

def validate_severe_response(response: str, query: str) -> dict:
    """
    Validate if response for severe queries contains required urgent information.
    Returns: {'valid': bool, 'missing': list of missing elements}
    """
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Only validate for severe queries
    severe_terms = ['severe', 'proliferative', 'PDR', 'severe NPDR']
    if not any(term.lower() in query_lower for term in severe_terms):
        return {'valid': True, 'missing': []}
    
    required_elements = {
        'referral': ['refer', 'referral', 'ophthalmologist', 'specialist', 'retina specialist'],
        'treatment': ['PRP', 'laser', 'anti-VEGF', 'photocoagulation', 'ranibizumab', 'aflibercept', 'bevacizumab'],
        'urgency': ['immediate', 'urgent', 'weeks', 'within', 'promptly', 'as soon as']
    }
    
    missing = []
    for category, keywords in required_elements.items():
        if not any(keyword.lower() in response_lower for keyword in keywords):
            missing.append(category)
    
    is_valid = len(missing) == 0
    
    return {'valid': is_valid, 'missing': missing}


def generate_response_with_validation(context: str, query: str, model: str = "mistral", max_retries: int = 2) -> str:
    """
    Generate response with validation and re-prompting if urgent info is missing.
    """
    for attempt in range(max_retries):
        response = generate_response(context, query, model)
        
        # Validate response
        validation = validate_severe_response(response, query)
        
        if validation['valid']:
            if attempt > 0:
                print(f"✓ Response validated after {attempt + 1} attempts")
            return response
        
        # If invalid and we have retries left, re-prompt with explicit requirements
        if attempt < max_retries - 1:
            print(f"⚠️ Response missing: {validation['missing']}. Re-prompting (attempt {attempt + 2}/{max_retries})...")
            
            # Create a more forceful prompt
            missing_str = ', '.join(validation['missing'])
            forced_query = f"{query}\n\nIMPORTANT: Your previous response was incomplete. You MUST include: {missing_str}. Extract this information from the context and list it first."
            
            response = generate_response(context, forced_query, model)
    
    # If still invalid after retries, return with warning
    print(f"⚠️ Final response still missing: {validation['missing']}")
    return response


def generate_response(context: str, query: str, model: str = "mistral") -> str:
    """
    Generate a response using the local Ollama LLM.
    Attempts the requested model first, falls back to 'tinyllama' if it fails.
    """
    
    # Detect if query is about severe/urgent conditions
    severe_terms = ['severe', 'proliferative', 'PDR', 'severe NPDR', 'advanced', 'urgent']
    is_severe_query = any(term.lower() in query.lower() for term in severe_terms)
    
    if is_severe_query:
        # STRICT prompt for severe cases - forces extraction of urgent info
        prompt = f"""
You are a medical assistant. The doctor is asking about a SEVERE/URGENT condition.

MANDATORY RESPONSE STRUCTURE - YOU MUST FOLLOW THIS EXACTLY:

1. FIRST LINE: State the most urgent action (e.g., "Immediate referral to ophthalmologist required")
2. SECOND: Specify the timeframe (e.g., "within 2-4 weeks")
3. THIRD: List the primary treatment options (e.g., "PRP laser therapy or anti-VEGF injections")
4. FOURTH: Mention monitoring frequency (e.g., "Follow-up every 3-6 months")
5. LAST: Any additional management (lifestyle, glycemic control)

❌ WRONG RESPONSE EXAMPLE (DO NOT DO THIS):
"Severe NPDR is a serious condition. Patients should maintain good blood sugar control, attend regular check-ups, and follow a healthy lifestyle. Ophthalmology referral may be considered."

✓ CORRECT RESPONSE EXAMPLE (DO THIS):
"1. Immediate referral to ophthalmologist/retina specialist required
2. Appointment within 2-4 weeks
3. Treatment options: PRP (panretinal photocoagulation) laser therapy OR anti-VEGF injections (ranibizumab, aflibercept)
4. Follow-up every 3-6 months after treatment
5. Continue glycemic control and blood pressure management"

DO NOT START WITH: general information, lifestyle advice, or background about the condition.
DO NOT DISCUSS: routine care, screening, or prevention BEFORE stating urgent actions.
IF YOU START WITH LIFESTYLE OR GENERAL INFO, YOU ARE DOING IT WRONG.

EXTRACT FROM CONTEXT:
- Any mention of "refer", "referral", "ophthalmologist", "specialist"
- Any mention of "PRP", "panretinal photocoagulation", "laser"
- Any mention of "anti-VEGF", drug names (ranibizumab, aflibercept, bevacizumab)
- Any mention of timeframes ("weeks", "months", "immediate")

Context:
{context}

Question: {query}

Answer (START WITH THE MOST URGENT ACTION):
1. """
    else:
        # Standard prompt for non-severe queries
        prompt = f"""
You are a specialized medical assistant helping a doctor with medical queries about diabetic retinopathy and related conditions.

CRITICAL INSTRUCTIONS:
1. ALWAYS prioritize urgent clinical actions first (e.g., "refer to ophthalmologist immediately")
2. For severe conditions (Severe NPDR, PDR, DME), mention specialist referral BEFORE other treatments
3. Provide specific timeframes (e.g., "within 2-4 weeks", "every 3-6 months monitoring")
4. List treatments in order of clinical priority: referral → laser (PRP) → anti-VEGF → monitoring
5. If the answer is NOT in the context, say "I don't know" - do NOT guess or hallucinate
6. Be specific about treatment names (e.g., "ranibizumab", "aflibercept", "bevacizumab")

RESPONSE FORMAT:
- Start with the most urgent action
- Use clear, structured language
- Include monitoring/follow-up recommendations
- Base everything on the provided context

Context:
{context}

Question: 
{query}

Answer (prioritize urgent actions first):"""

    # Helper to call the Ollama API
    def call_ollama(model_name):
        try:
            print(f"Calling Ollama with model: {model_name}")
            print(f"Prompt length: {len(prompt)} characters")
            response = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            result = response['message']['content']
            print(f"Successfully generated response with {model_name} ({len(result)} chars)")
            return result
        except Exception as e:
            print(f"Error calling Ollama with '{model_name}': {e}")
            return None

    # Try using the main model first
    print(f"Primary model attempt: {model}")
    response = call_ollama(model)
    if response:
        return response
    
    # Fallback to tinyllama if the primary model fails
    if model != "tinyllama":
        print("⚠️ Primary model failed. Attempting fallback to 'tinyllama'...")
        response = call_ollama("tinyllama")
        if response:
            print("✓ Fallback to tinyllama successful")
            return response
        else:
            print("✗ Fallback to tinyllama also failed")

    # If all attempts fail, return a helpful error message
    return "I apologize, but I encountered an error generating the response. Please try again later."