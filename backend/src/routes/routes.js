const express = require('express');
const multer = require('multer');
const { uploadResume } = require('../controllers/controller');

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

router.post('/upload-resume', upload.single('resume'), uploadResume);



module.exports = router;
