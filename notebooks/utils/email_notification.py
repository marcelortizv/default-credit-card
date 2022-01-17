import smtplib, ssl
import settings_email as cfg


def send_email(list_to, subject, body):

    port = 465
    smtp_server = "smtp.gmail.com"
    sender_email = cfg.email
    password = cfg.password
    for receiver_email in list_to:
        message = f"""\
Subject: {subject}
{body}"""
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)