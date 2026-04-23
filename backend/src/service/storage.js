const { v2: cloudinary } = require('cloudinary');

cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET
});

function uploadFile(file) {
    return new Promise((resolve,reject)=>{
        if (!file || !file.buffer) {
            return reject(new Error('File buffer is missing.'));
        }

        const base64File = `data:${file.mimetype};base64,${file.buffer.toString('base64')}`;
        cloudinary.uploader.upload(
            base64File,
            {
                resource_type: 'raw',
                folder: 'resumes',
                public_id: file.originalname,
                access_model: 'public'
            }
        )
            .then(resolve)
            .catch(reject);
    });
}

module.exports= uploadFile


