const fs = require('fs');
const { spawn } = require('child_process');

fs.readFile('images/1.jpg','', (err, data) => {
    if(!err){
        const child = spawn('python',['experiment/experiment.py'])

        child.stdin.write("hi\n");

        
        setTimeout(function(){
           child.stdin.write("world\n");
        },2000)
        
        
        
        child.stdout.on('data', data => {
            console.log(data.toString());
        });
    }else{
        console.error(err)
    }
})