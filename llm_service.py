from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import json
import re
from datetime import datetime

def process_transcript(transcript: str, max_tokens=4000):
    try:
        # Check API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ API key not found")
            return _fallback_processing(transcript)

        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=0.3,
            model_name="mixtral-8x7b-32768",
            groq_api_key=groq_api_key,
            max_tokens=max_tokens
        )

        # Advanced Prompt Template
        prompt = PromptTemplate(
            input_variables=["transcript"],
            template="""
            Perform a comprehensive analysis of the following meeting transcript:

            TRANSCRIPT:
            {transcript}

            ANALYSIS REQUIREMENTS:
            1. Detailed Meeting Summary
               - Key discussion points
               - Tone and sentiment
               - Overall objectives

            2. Actionable CRM Insights
               a) Identified Pain Points
               b) Client/Stakeholder Objections
               c) Proposed Resolutions
               d) Concrete Action Items

            3. Strategic Recommendations
               - Immediate next steps
               - Long-term strategic implications

            OUTPUT FORMAT: Structured JSON with clear, professional insights
            """
        )

        # Process transcript
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(transcript=transcript)

        # Clean and parse JSON
        result = _clean_json_response(result)
        parsed_result = json.loads(result)

        # Add metadata
        parsed_result['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'transcript_length': len(transcript)
        }

        return parsed_result

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return _fallback_processing(transcript)

def _clean_json_response(response):
    # Remove code blocks and clean JSON
    response = response.replace('```json', '').replace('```', '').strip()
    response = re.sub(r',\s*}', '}', response)
    return response

def _fallback_processing(transcript):
    # Basic fallback processing
    words = transcript.split()
    return {
        'summary': {
            'summary': f"Meeting transcript with {len(words)} words",
            'discussion_topics': ['General Discussion']
        },
        'crm_insights': {
            'pain_points': ['Detailed analysis requires full processing'],
            'objections': ['Comprehensive review needed'],
            'resolutions': ['Manual review recommended'],
            'action_items': ['Review full transcript']
        }
    }