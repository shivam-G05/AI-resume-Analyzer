const express = require('express');
require('dotenv').config();
const cors = require('cors');
const routes = require('./routes/routes');

const app = express();

app.use(express.json());

app.use(cors({
    origin:"http://localhost:5173"
}));


app.get('/', (req, res) => {
    res.status(200).json({ message: 'Backend is running' });
});

app.use('/api', routes);

module.exports = app;
