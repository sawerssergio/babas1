import express from 'express';
import productsRouter from './routes/products.js';
import salesRouter from './routes/sales.js';
import usersRouter from './routes/users.js';

const app = express();
app.use(express.json());
app.use('/api/products', productsRouter);
app.use('/api/sales', salesRouter);
app.use('/api/users', usersRouter);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
