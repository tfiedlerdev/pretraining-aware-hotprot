import requests
from urllib.parse import quote
import configparser
import os


class TelegramBot:
    def __init__(self):
        configFile = "config.ini"
        if not os.path.exists(configFile):
            with open(configFile, "w") as f:
                f.write(
                    "[telegram]\nChatId = [TELEGRAM_CHAT_ID]\nBotKey = [TELEGRAM_BOT_KEY]"
                )
            raise Exception(
                f"For using the telegram bot, add the telegram chat id and bot key to the newly created {configFile} config file."
            )

        self.config = configparser.ConfigParser()
        self.config.read(configFile)

        self.enabled = True

        self.chatId = self.config["telegram"]["ChatId"]
        self.botKey = self.config["telegram"]["BotKey"]
        if "[" in self.chatId or "[" in self.botKey:
            raise Exception("Invalid format of ChatId or BotKey field in {configFile}")

    def send_photo(self, imagePath: str, caption: str = None) -> None:
        if self.enabled:
            self._fetch_send_photo(imagePath, caption)

    def send_telegram(self, message: str) -> None:
        if not self.enabled:
            return {"result": {"message_id": -1}}
        return self._fetch_send_message(message)

    def edit_text_message(self, message_id: int, editedText: str) -> None:
        if not self.enabled:
            return
        return self._fetch_edit_message(message_id, editedText)

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
            return response
        except Exception:
            return False

    def _fetch_edit_message(self, messageId: int, editedText: str):
        url = (
            "https://api.telegram.org/bot"
            + self.botKey
            + "/editMessageText?chat_id="
            + self.chatId
            + "&message_id="
            + str(messageId)
            + "&text="
            + quote(editedText)
        )

        try:
            response = (requests.get(url)).json()

            return response
        except Exception:
            print("Exception while decodin response from editing telegram message")
            return False

    def _fetch_send_photo(self, filePath: str, caption: str = None) -> bool:
        url = (
            "https://api.telegram.org/bot"
            + self.botKey
            + "/sendPhoto?chat_id="
            + self.chatId
        )

        if caption is not None:
            url += "&caption=" + quote(caption)

        try:
            response = (
                requests.post(url, files=dict(photo=open(filePath, "rb")))
            ).json()
            return response["ok"]
        except Exception:
            return False
