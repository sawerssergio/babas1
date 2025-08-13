import { Router } from 'express';
import { db } from '../storage.js';

const router = Router();

router.get('/', (req, res) => {
  res.json(db.sales);
});

router.post('/', (req, res) => {
  const { productId, quantity } = req.body;
  const product = db.products.find(p => p.id === Number(productId));
  if (!product) return res.status(404).json({ error: 'Product not found' });
  const qty = Number(quantity) || 0;
  if (product.stock < qty) return res.status(400).json({ error: 'Insufficient stock' });
  product.stock -= qty;
  const sale = { id: db.nextSaleId++, productId: product.id, quantity: qty, date: new Date().toISOString() };
  db.sales.push(sale);
  res.status(201).json(sale);
});

export default router;
