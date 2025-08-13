import { Router } from 'express';
import { db } from '../storage.js';

const router = Router();

router.get('/', (req, res) => {
  res.json(db.products);
});

router.post('/', (req, res) => {
  const { name, stock } = req.body;
  const product = { id: db.nextProductId++, name, stock: Number(stock) || 0 };
  db.products.push(product);
  res.status(201).json(product);
});

router.put('/:id', (req, res) => {
  const id = Number(req.params.id);
  const product = db.products.find(p => p.id === id);
  if (!product) return res.status(404).json({ error: 'Product not found' });
  const { name, stock } = req.body;
  if (name !== undefined) product.name = name;
  if (stock !== undefined) product.stock = Number(stock);
  res.json(product);
});

router.delete('/:id', (req, res) => {
  const id = Number(req.params.id);
  const index = db.products.findIndex(p => p.id === id);
  if (index === -1) return res.status(404).json({ error: 'Product not found' });
  const [removed] = db.products.splice(index, 1);
  res.json(removed);
});

export default router;
