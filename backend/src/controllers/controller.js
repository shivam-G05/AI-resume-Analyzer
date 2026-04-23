const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const axios = require('axios');
function runPythonParser(buffer, filename) {
    return new Promise((resolve, reject) => {

        
        const ext = path.extname(filename);
        const tmpPath = path.join(os.tmpdir(), `resume_${Date.now()}${ext}`);
        fs.writeFileSync(tmpPath, buffer);

        const pythonBin = process.env.PYTHON_BIN || 'python';
        const scriptPath = path.resolve(__dirname, '../../../process_url_with_python.py');

        const proc = spawn(
            pythonBin,
            [scriptPath, tmpPath],   
            { timeout: 120000 }
        );

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
        proc.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

        proc.on('close', (code) => {
            try { fs.unlinkSync(tmpPath); } catch (_) {}   // always clean up

            console.log('Python exit code:', code);
            console.log('Python stderr:', stderr);
            console.log('Python stdout:', stdout);

            if (code !== 0) {
                return reject(new Error(stderr || `Python exited with code ${code}`));
            }
            try {
                resolve(JSON.parse(stdout));
            } catch (parseError) {
                reject(new Error(`Invalid JSON: ${parseError.message}\nOutput: ${stdout}`));
            }
        });

        proc.on('error', (err) => {
            try { fs.unlinkSync(tmpPath); } catch (_) {}
            reject(new Error(`Failed to start python: ${err.message}`));
        });
    });
}

async function uploadResume(req, res) {
    try {
        if (!req.file) {
            return res.status(400).json({
                message: 'No file received. Send file with form-data key "resume".'
            });
        }

        const parsedResume = await runPythonParser(req.file.buffer, req.file.originalname);

        return res.status(200).json({
            message: 'Resume parsed successfully',
            parsedResume
        });

    } catch (error) {
        return res.status(500).json({
            message: 'Error processing resume',
            error: error.message
        });
    }
}



module.exports = { uploadResume };