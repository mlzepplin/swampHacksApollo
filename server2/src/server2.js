/** Imports **/

const WebSocket = require('ws');
const fs = require('fs');
const path = require('path')
const { spawn } = require('child_process');



/** Clear Old Images **/

console.log('Clearing images...');
let dirToDelete = './images';
fs.readdir(dirToDelete, (err, files) => {
    if (err) throw err;

    for (const file of files) {
        fs.unlink(path.join(dirToDelete, file), err => {
            if (err) throw err;
        });
    }
});
console.log('Done\n');



/** Initialize Stuff **/

console.log('Starting WS Server...');
const wss = new WebSocket.Server({ port: 8080 });
console.log('Done\n');

console.log('Starting Worker...');
const worker = spawn('python', ['src/worker.py'])
worker.stdout.on('data', data => {
    console.log('Worker|>',data.toString());
});
console.log('Done\n');

console.log('\nAll Systems Go! Ready for incoming connections...');



/** Main Logic **/

let id = 0;

wss.on('connection', ws => {
    console.log("\nNEW CONNECTION!!!!")

    ws.send('pong');

    ws.on('message', message => {
        console.log('\n\n');
        console.log(id, "Got Message")

        id++;

        console.log('Sending work to worker...');
        worker.stdin.write(message+"\n");
        console.log("Done\n")

        ws.send('69')

        // console.log('Starting Image Write...');
        // fs.writeFile(`./images/${id}.jpg`, message, err => {
        //     console.log("Done")
            
        // })
    });

   
});