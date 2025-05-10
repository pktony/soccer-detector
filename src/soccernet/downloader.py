# pip install SoccerNet --upgrade

from SoccerNet.Downloader import SoccerNetDownloader

def download_action_spotting(localPath):
    mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=localPath)

    mySoccerNetDownloader.downloadGames(files=["Labels-v3.json", "Frames-v3.zip"], split=["train","valid"], task="frames")


def download_tracking(localPath):
    mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=localPath)

    mySoccerNetDownloader.downloadDataTask(split=["train","valid", "challage"], task="tracking")