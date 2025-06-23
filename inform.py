import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


receiver='1124998618@qq.com'
sender = '1302905387@qq.com'
subject = f'训练已完成{datetime.now().strftime("%H:%M:%S.%f")}'
message = f'训练已完成'

try:
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    with smtplib.SMTP('smtp.qq.com', 587) as server:
        server.starttls()
        server.login(sender, 'vyfjlrupoebojfdh')
        server.sendmail(sender, receiver, msg.as_string())
except smtplib.SMTPResponseException as e:
    if e.smtp_code == -1 and e.smtp_error == b'\x00\x00\x00':
        # 忽略特定异常
        pass
    else:
        error_msg = f'邮件发送失败 (SMTP 错误): {e},当前时间为 {datetime.now().strftime("%H:%M:%S.%f")}'
except Exception as e:
    error_msg = f'邮件发送失败 (其他错误): {e},当前时间为 {datetime.now().strftime("%H:%M:%S.%f")}'
