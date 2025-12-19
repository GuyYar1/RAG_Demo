import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

# Clinical treatment guidelines for Severe NPDR (Level 3/4)
SEVERE_NPDR_FACTS = """
DIAGNOSIS: Severe Nonproliferative Diabetic Retinopathy (Severe NPDR)
RISK: Nearly 50% risk of progressing to high-risk Proliferative Diabetic Retinopathy (PDR) within 1 year

P1 - PRIMARY ACTION (Referral):
Promptly refer to an ophthalmologist knowledgeable and experienced in diabetic retinopathy management

P2 - INTERVENTION OPTIONS (Treatment):
1. Panretinal photocoagulation (PRP) surgery - remains an important treatment for PDR and severe NPDR
2. Intravitreal anti-vascular endothelial growth factor (anti-VEGF) agents - reduce the severity of DR

P3 - EXPEDITED MONITORING:
Hospital eye services should monitor disease progression every 3-6 months for severe or very severe NPDR not currently treated

P4 - GENERAL CARE (Secondary - subordinate to ophthalmic care):
See endocrinologist for HbA1c management and systemic diabetes control
""" 

def validate_severe_response(response: str, query: str, profile) -> dict:
    """
    Validate if response for severe queries contains required urgent information.
    Returns: {'valid': bool, 'missing': list of missing elements}
    """

    if not response:
        logger.warning(f"Validation failed: No response provided for query: {query}")
        return {'valid': False, 'missing': ['response']}

    response_lower = response.lower()
    query_lower = query.lower()
 
    from profile_extractor import is_severe_condition
    
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Only validate for severe queries
    if not is_severe_condition(profile):
        return {'valid': True, 'missing': []}
    
    # Stricter validation - check for specific clinical facts
    required_elements = {
        'diagnosis': ['severe NPDR', 'severe nonproliferative', 'nonproliferative diabetic retinopathy', 'severe diabetic retinopathy'],
        'risk': ['50%', '50 percent', 'progression', 'PDR', 'proliferative'],
        'referral': ['ophthalmologist', 'promptly refer', 'prompt referral', 'experienced in diabetic retinopathy', 'knowledgeable'],
        'treatment': ['PRP', 'panretinal photocoagulation', 'anti-VEGF', 'intravitreal', 'anti-vascular'],
        'monitoring': ['3-6 months', 'every 3', '3 to 6', 'hospital eye services', 'monitor disease progression']
    }
    
    missing = []
    for category, keywords in required_elements.items():
        if not any(keyword.lower() in response_lower for keyword in keywords):
            missing.append(category)
    
    is_valid = len(missing) == 0
    
    return {'valid': is_valid, 'missing': missing}
    

def generate_response_with_validation(context: str, query: str, profile, model: str = "mistral", max_retries: int = 2) -> str:
    """
    Generate response with validation and re-prompting if urgent info is missing.
    """
    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt + 1}: Generating response...")
        response = generate_response(context, query, profile, model)
        
        # Validate response
        if response is None:
            logger.warning(" Response is None. Skipping validation.")
            return "I apologize, but I encountered an error generating the response. Please try again later."

        validation = validate_severe_response(response, query, profile)
        
        if validation['valid']:
            if attempt > 0:
                print(f"✓ Response validated after {attempt + 1} attempts")
            return response
        
        # If invalid and we have retries left, re-prompt with explicit requirements
        if attempt < max_retries - 1:
            print(f"⚠️ Response missing: {validation['missing']}. Re-prompting (attempt {attempt + 2}/{max_retries})...")
            
            # Create a more forceful prompt
            missing_str = ', '.join(validation['missing'])
            forced_query = f"{query}\n\nCRITICAL: Your previous response was incomplete. You MUST include: {missing_str}. Extract this information from the context and place it at the very beginning of your response."
            
            response = generate_response(context, forced_query, profile, model)
    
    # If still invalid after retries, return with warning
    print(f"⚠️ Final response still missing: {validation['missing']}")
    return response

def generate_response(context: str, query: str, profile, model: str = "mistral") -> str:
    """
    Generate a response using the local Ollama LLM with user profile integration.
    Attempts the requested model first, falls back to 'tinyllama' if it fails.
    """
    from profile_extractor import is_severe_condition
    
    # Detect if query is about severe/urgent conditions
    is_severe = is_severe_condition(profile)
    
    # Build profile context
    profile_context = ""
    if profile.age:
        profile_context += f"Patient age: {profile.age} years old\n"
    if profile.gender:
        profile_context += f"Gender: {profile.gender}\n"
    if profile.medical_history:
        profile_context += f"Medical history: {', '.join(profile.medical_history)}\n"
    if profile.severity_level:
        profile_context += f"Severity level: {profile.severity_level}/4\n"
    if profile.condition:
        profile_context += f"Diagnosed condition: {profile.condition}\n"
    
    if is_severe:
        # STRICT prompt for severe cases - enforces P1-P5 clinical hierarchy
        prompt = f"""
You are a medical assistant. The doctor is asking about a SEVERE/URGENT condition for a specific patient.

PATIENT PROFILE:
{profile_context}

MANDATORY CLINICAL HIERARCHY - YOU MUST FOLLOW THIS P1-P5 STRUCTURE EXACTLY:

**P1 - DIAGNOSIS & RISK** (State the condition and its progression risk FIRST):
For Level 3/4 severity: "This is Severe Nonproliferative Diabetic Retinopathy (Severe NPDR) with nearly 50% risk of progressing to Proliferative Diabetic Retinopathy (PDR) within 1 year"

**P2 - PRIMARY ACTION (Referral - HIGHEST PRIORITY)**:
"Promptly refer to an ophthalmologist knowledgeable and experienced in diabetic retinopathy management"

**P3 - INTERVENTION OPTIONS (Treatment - SECOND PRIORITY)**:
List BOTH treatments specifically:
1. "Panretinal photocoagulation (PRP) surgery - remains an important treatment for PDR and severe NPDR"
2. "Intravitreal anti-vascular endothelial growth factor (anti-VEGF) agents - reduce the severity of DR"

**P4 - EXPEDITED MONITORING (THIRD PRIORITY)**:
"Hospital eye services should monitor disease progression every 3-6 months for severe NPDR not currently treated"

**P5 - GENERAL CARE (SUBORDINATE - LAST PRIORITY)**:
Endocrinologist for HbA1c management, blood pressure control, lifestyle (diet, exercise)
DO NOT mention P5 items before completing P1-P4.

❌ WRONG RESPONSE (DO NOT DO THIS):
"Maintain good blood sugar control, see your doctor regularly, and follow a healthy lifestyle. You should also see an ophthalmologist."

✓ CORRECT RESPONSE (DO THIS):
"**DIAGNOSIS**: Severe Nonproliferative Diabetic Retinopathy (Severe NPDR)
**RISK**: Nearly 50% risk of progressing to PDR within 1 year

**P1 - PRIMARY ACTION**: Promptly refer to an ophthalmologist knowledgeable and experienced in diabetic retinopathy management

**P2 - TREATMENT OPTIONS**:
1. Panretinal photocoagulation (PRP) surgery
2. Intravitreal anti-VEGF agents

**P3 - MONITORING**: Hospital eye services every 3-6 months

**P4 - GENERAL CARE**: See endocrinologist for HbA1c management, maintain blood pressure control"

CRITICAL RULES:
- START with P1 (Diagnosis & Risk), NOT with lifestyle advice
- P2 (Referral) must come BEFORE P5 (general care)
- IF YOU START WITH "control blood sugar" or "see your doctor", YOU HAVE FAILED
- DO NOT say "regular check-ups" - say "every 3-6 months"

REQUIRED CLINICAL FACTS FROM GUIDELINES:
{SEVERE_NPDR_FACTS}

YOU MUST INCLUDE THESE FACTS IN YOUR RESPONSE:
1. Diagnosis: Severe NPDR
2. Risk: 50% progression to PDR within 1 year
3. Referral: Ophthalmologist experienced in diabetic retinopathy
4. Treatments: PRP surgery AND anti-VEGF (name both)
5. Monitoring: Every 3-6 months at hospital eye services

Medical Context:
{context}

Question: {query}

Answer (FOLLOW P1-P2-P3-P4-P5 HIERARCHY - START WITH DIAGNOSIS):
"""
    else:
        # Standard prompt for non-severe queries with profile awareness
        prompt = f"""
You are a specialized medical assistant helping a doctor with medical queries about diabetic retinopathy and related conditions.

PATIENT PROFILE:
{profile_context}

CRITICAL INSTRUCTIONS:
1. ALWAYS prioritize urgent clinical actions first (e.g., "refer to ophthalmologist immediately")
2. For severe conditions (Severe NPDR, PDR, DME), mention specialist referral BEFORE other treatments
3. Provide specific timeframes (e.g., "within 2-4 weeks", "every 3-6 months monitoring")
4. List treatments in order of clinical priority: referral → laser (PRP) → anti-VEGF → monitoring
5. Tailor lifestyle and management advice to the patient's age ({profile.age if profile.age else 'adult'}) and medical history
6. If the answer is NOT in the context, say "I don't know" - do NOT guess or hallucinate
7. Be specific about treatment names (e.g., "ranibizumab", "aflibercept", "bevacizumab")

RESPONSE FORMAT:
- Start with the most urgent action (if applicable)
- Use clear, structured language
- Include monitoring/follow-up recommendations
- Provide age-appropriate lifestyle advice
- Base everything on the provided context and patient profile

Medical Context:
{context}

Question: 
{query}

Answer (prioritize urgent actions first, then tailor advice to patient profile):
"""

    logger.info("ENTER generate_response core")
    logger.info(f"is_severe={is_severe}, model={model}")
    result = call_groq_api(model, prompt, profile)
    logger.info(f"Result from call_groq_api is None? {result is None}")
    return result

   

# Helper to call the groq Inference API
def call_groq_api(model_name, prompt, profile):
    try:
        logger.info(f"Calling Groq API with model: {model_name}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Profile: {profile}")
        
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY not set!")
            return None
        
        # Initialize Groq client
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Map model names to Groq models
        groq_model_map = {
            #"mistral": "mixtral-8x7b-32768",  # Fast Mixtral but groq has been decommissioned it and is no longer supported
            # Main fast general model

            "mistral": "llama-3.3-70b-versatile",   # main high-quality model
            "tinyllama": "llama-3.1-8b-instant",    # cheaper/faster fallback
            "biomistral": "llama-3.3-70b-versatile",
            #"biomistral": "mixtral-8x7b-32768"            
        }
        
        groq_model = groq_model_map.get(model_name, groq_model_map["biomistral"])
        
        # Call Groq API
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.1,
            top_p=0.8
        )
        
        result = response.choices[0].message.content
        logger.info(f"Successfully generated response with {groq_model} ({len(result)} chars)")
        return result
        
    except Exception as e:
        import traceback
        logger.error("### GROQ ERROR ###")
        logger.error(e)
        logger.error(traceback.format_exc())
        logger.error(f"Error calling Groq API with '{model_name}': {e}")
        logger.error(traceback.format_exc())
        return None

# Try using the main model first
def generate_response_with_retry(context, query, profile, model="mistral"):
    logger.info(f"Primary model attempt: {model}")
    response = call_groq_api(model, context, profile)
    if response is None:
        logger.warning(f"Response from {model} is None. Retrying with fallback model.")
    if response:
        return response
    
    # Fallback to tinyllama if the primary model fails
    if model == "mistral" and response is None:
        logger.warning("model (mistral) failed. Attempting fallback to 'tinyllama'...")
        
        response = call_groq_api("tinyllama", context, profile)
        if response:
            logger.info("✓ Fallback to tinyllama successful")
            return response
        else:
            logger.error("✗ Fallback to tinyllama also failed")

    # If all attempts fail, return a helpful error message
    return "I apologize, but I encountered an error generating the response. Please try again later."


