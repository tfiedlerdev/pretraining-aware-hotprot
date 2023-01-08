import math
import requests
from urllib.parse import quote
import configparser
import os



class TelegramBot():

    def __init__(self):
        configFile = 'config.ini'
        if not os.path.exists(configFile):
            with open(configFile, "w") as f: 
                f.write("[telegram]\nChatId = [TELEGRAM_CHAT_ID]\nBotKey = [TELEGRAM_BOT_KEY]")
            raise Exception(f"For using the telegram bot, add the telegram chat id and bot key to the newly created {configFile} config file.")

        self.config = configparser.ConfigParser()
        self.config.read(configFile)

        self.enabled = True

        self.chatId = self.config["ChatId"]
        self.botKey = self.config["BotKey"]
        if "[" in self.chatId or "[" in self.botKey:
            raise Exception("Invalid format of ChatId or BotKey field in {configFile}")

    def send_photo(self,imagePath: str, caption: str = None) -> None:
        if self.enabled:
            self._fetch_send_photo(imagePath, caption)

    def send_telegram(self,message: str) -> None:
        if not self.enabled:
            return
        packages_remaining = [message]
        max_messages_num = 40
        while len(packages_remaining) > 0 and max_messages_num > 0:
            curr_package = packages_remaining.pop(0)
            message_sent = self._fetch_send_message(curr_package)
            if message_sent:
                max_messages_num -= 1
            if not message_sent:
                if len(curr_package) < 10:
                    self._fetch_send_message("Telegram failed")
                    break
                num_of_chars_first = math.ceil(len(curr_package) / 2)
                first_package = curr_package[0:num_of_chars_first]
                second_package = curr_package[num_of_chars_first : len(curr_package)]

                packages_remaining.insert(0, second_package)
                packages_remaining.insert(0, first_package)
        if max_messages_num == 0:
            self._fetch_send_message("Sending failed. Too many messages sent.")


    def _fetch_send_message(self, message: str) -> bool:
        message = str(message)
        url = (
            "https://api.telegram.org/bot"
            + self.botKey
            + "/sendMessage?chat_id="
            + self.chatId
            + "&text="
            + quote(message)
        )

        try:
            response = (requests.get(url)).json()
            return response["ok"]
        except:
            return False

    def _fetch_send_photo(self,filePath: str, caption: str = None) -> bool:
        url = "https://api.telegram.org/bot" + self.botKey+ "/sendPhoto?chat_id=" + self.chatId
        
        if caption != None:
            url += "&caption="+ quote(caption)

        try:
            response = (requests.post(url, files=dict(photo=open(filePath, "rb")))).json()
            return response["ok"]
        except:
            return False