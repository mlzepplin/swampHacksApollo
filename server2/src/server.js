const WebSocket = require('ws');
const fs = require('fs');
const path = require('path')

const wss = new WebSocket.Server({ port: 8080 });

console.log("Server Ready...")

let id = 0;

//clear all images
let dirToDelete = './images';
fs.readdir(dirToDelete, (err, files) => {
    if (err) throw err;
  
    for (const file of files) {
      fs.unlink(path.join(dirToDelete, file), err => {
        if (err) throw err;
      });
    }
  });

wss.on('connection', ws => {
    console.log("\nNEW CONNECTION!!!!")

  ws.on('message', message => {
    //console.log('received: ', message.substring(0, 100));
    console.log(id, "got message")
    id++;
    fs.writeFileSync(`./images/${id}.jpg`, message)
  });

  ws.send('pong');
});