import smtplib
import socket


def email_with_gmail(username, password,
                     to, subject, body):
    '''
    Sends an email from an gmail account.

    Parameters
    ----------
    username, password: string
        Gmail username and password: the prefix before @gmail.com
    to: string
        Email address of the recipient.
    subject, body: string
        Email subject and content.

    Returns
    -------
    None
    '''
    headers = '\r\n'.join([
        'from: {}'.format(username),
        'subject: {}'.format(subject),
        'to: {}'.format(to),
        'mime-version: 1.0',
        'content-type: text/html'])

    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.ehlo()
    session.starttls()
    session.login(username, password)
    session.sendmail(username, to, headers + '\r\n\r\n' + body)


def get_local_ip():
    '''
    Returns the local IP address.

    Parameters
    ----------
    None

    Returns
    -------
    IP: string
        IP address
    '''
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("gmail.com", 80))
        return s.getsockname()[0]
    finally:
        s.close()
