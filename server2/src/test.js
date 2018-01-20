const WebSocket = require('ws');

const ws = new WebSocket('ws://10.192.134.211:8080');

ws.on('open', () => {
  ws.send('ping');

  ws.on('message', message => {
    console.log(message);
  });
});