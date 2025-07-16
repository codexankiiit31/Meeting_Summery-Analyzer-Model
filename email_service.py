# services/email_service.py
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def send_email_summary(to_email: str, subject: str, body: str) -> bool:
    """
    Sends an email with the meeting summary using more secure SMTP configuration.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body text
        
    Returns:
        Boolean indicating success or failure
    """
    # Get email credentials from environment variables
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))  # Changed to SSL port
    
    # Comprehensive credential validation
    if not sender_email:
        logger.error("Sender email not configured in .env file.")
        return False
    
    if not sender_password:
        logger.error("Sender password not configured in .env file.")
        return False
    
    try:
        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = to_email
        message["Subject"] = subject
        
        # Add timestamp to the body
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_body = f"{body}\n\nGenerated on: {timestamp}"
        
        # Attach body text
        message.attach(MIMEText(full_body, "plain"))
        
        # Create a secure SSL context
        context = ssl.create_default_context()
        
        # Use SSL instead of TLS for more reliable email sending
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            # Increase timeout to prevent premature disconnection
            server.timeout = 30
            
            # Login to the email server
            server.login(sender_email, sender_password)
            
            # Send email
            server.sendmail(
                sender_email, 
                to_email, 
                message.as_string()
            )
            
        logger.info(f"Email sent successfully to {to_email}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        logger.error(f"SMTP Authentication Error: Failed to login to {smtp_server}.")
        logger.error("Possible reasons:")
        logger.error("1. Incorrect email or password")
        logger.error("2. Less secure app access might be disabled")
        logger.error("3. For Gmail, use an App Password")
        logger.error("   - Go to Google Account > Security > App Passwords")
        logger.error("   - Generate a specific app password for your application")
        return False
    
    except smtplib.SMTPException as smtp_error:
        logger.error(f"SMTP Error: {smtp_error}")
        return False
    
    except ssl.SSLError as ssl_error:
        logger.error(f"SSL Connection Error: {ssl_error}")
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error when sending email: {e}")
        return False

def format_meeting_summary_email(meeting_data: dict) -> tuple:
    """
    Formats meeting data into an email subject and body.
    
    Args:
        meeting_data: Dictionary containing meeting summary and insights
        
    Returns:
        Tuple of (subject, body)
    """
    try:
        # Extract meeting information
        summary = meeting_data.get('summary', {}).get('summary', 'No summary available.')
        discussion_topics = meeting_data.get('summary', {}).get('discussion_topics', [])
        
        # Extract CRM insights
        pain_points = meeting_data.get('crm_insights', {}).get('pain_points', [])
        objections = meeting_data.get('crm_insights', {}).get('objections', [])
        resolutions = meeting_data.get('crm_insights', {}).get('resolutions', [])
        action_items = meeting_data.get('crm_insights', {}).get('action_items', [])
        
        # Create email subject
        subject_topic = discussion_topics[0] if discussion_topics else "Meeting"
        subject = f"Meeting Summary: {subject_topic}"
        
        # Create email body with nice formatting
        body = f"""MEETING SUMMARY REPORT
{'=' * 50}

SUMMARY:
{summary}

DISCUSSION TOPICS:
{chr(10).join(['• ' + str(topic) for topic in discussion_topics]) or '• No topics discussed'}

CLIENT PAIN POINTS:
{chr(10).join(['• ' + str(pp) for pp in pain_points]) or '• No pain points identified'}

OBJECTIONS:
{chr(10).join(['• ' + str(obj) for obj in objections]) or '• No objections raised'}

PROPOSED RESOLUTIONS:
{chr(10).join(['• ' + str(res) for res in resolutions]) or '• No resolutions discussed'}

ACTION ITEMS:
{chr(10).join(['• ' + str(ai) for ai in action_items]) or '• No action items defined'}

{'=' * 50}
Meeting ID: {meeting_data.get('meeting_id', 'Not available')}

This summary was generated by the AI Meeting Summary + CRM Note Generator.
"""
        
        return subject, body
    
    except Exception as e:
        logger.error(f"Error formatting email: {e}")
        return "Meeting Summary", "Unable to generate detailed summary."

# Test function when script is run directly
if __name__ == "__main__":
    # Test email sending
    test_email = input("Enter test email address: ")
    test_meeting_data = {
        'summary': {
            'summary': 'Test meeting summary',
            'discussion_topics': ['Topic 1', 'Topic 2']
        },
        'crm_insights': {
            'pain_points': ['Pain Point 1'],
            'objections': ['Objection 1'],
            'resolutions': ['Resolution 1'],
            'action_items': ['Action Item 1']
        },
        'meeting_id': 'test-meeting-123'
    }
    
    subject, body = format_meeting_summary_email(test_meeting_data)
    result = send_email_summary(test_email, subject, body)
    print(f"Email sending result: {'Success' if result else 'Failed'}")