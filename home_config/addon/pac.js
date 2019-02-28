// pac file

domains = {
    'google.com': 1,
    'gmail.com': 1,
    'googleapis.com': 1,
    'google.com.hk': 1,
    'gstatic.com': 1,
    'googleusercontent.com': 1,
    'facebook.com': 1,
    'facebook.net': 1,
    'twitter.com': 1,
    'twimg.com': 1,
    'youtube.com': 1,
    'ytimg.com': 1,
    'ggpht.com': 1,
    'googlevideo.com': 1,
    'youtube-nocookie.com': 1,
    'twitch.tv': 1,
    'onedrive.live.com': 1,
    'zh.wikipedia.org': 1,
    'tumblr.com': 1,
    'bandainamcoentstore.com': 1,
    'pepitastore.com': 1,
    't.co': 1,
    'appspot.com': 1,
    'soundcloud.com': 1,
    // 'indiegala.com': 1,
    // 'bundlestars.com': 1,
    // 'groupees.com': 1,
    // 'dlh.net': 1,
    // 'hrkgame.com': 1,
    // 'stackoverflow.com': 1,
    'steamcommunity.com': 1
};

var proxy = 'SOCKS5 127.0.0.1:1080; SOCKS 127.0.0.1:1080';

var direct = 'DIRECT;';

var hasOwnProperty = Object.hasOwnProperty;

function FindProxyForURL(url, host) {
    var suffix;
    var pos = host.lastIndexOf('.');
    pos = host.lastIndexOf('.', pos - 1);
    while(1) {
        if (pos <= 0) {
            if (hasOwnProperty.call(domains, host)) {
                return proxy;
            } else {
                return direct;
            }
        }
        suffix = host.substring(pos + 1);
        if (hasOwnProperty.call(domains, suffix)) {
            return proxy;
        }
        pos = host.lastIndexOf('.', pos - 1);
    }
}
