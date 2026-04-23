const express = require('express');
require('dotenv').config();
const cors = require('cors');
const routes = require('./routes/routes');

const app = express();

app.use(express.json());

app.use(cors({
    origin:"https://ai-resume-analyzer-vpqu.onrender.com"
}));


app.get('/', (req, res) => {
    res.status(200).json({ message: 'Backend is running' });
});

app.use('/api', routes);

module.exports = app;
