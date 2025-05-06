# pip install SoccerNet --upgrade

from SoccerNet.Downloader import SoccerNetDownloader

def download_soccerNet(localPath):
    mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=localPath)

    mySoccerNetDownloader.downloadGames(files=["Labels-v3.json", "Frames-v3.zip"], split=["train","valid","test"], task="frames")