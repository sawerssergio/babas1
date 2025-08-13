import { db } from '../storage.js';
import { parseJson, sendJson } from '../utils.js';

export async function handleSales(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === 'GET' && url.pathname === '/api/sales') {
    sendJson(res, db.sales);
  } else if (req.method === 'POST' && url.pathname === '/api/sales') {
    const { productId, quantity } = await parseJson(req);
    const product = db.products.find(p => p.id === Number(productId));
    if (!product) return sendJson(res, { error: 'Product not found' }, 404);
    const qty = Number(quantity) || 0;
    if (product.stock < qty) return sendJson(res, { error: 'Insufficient stock' }, 400);
    product.stock -= qty;
    const sale = { id: db.nextSaleId++, productId: product.id, quantity: qty, date: new Date().toISOString() };
    db.sales.push(sale);
    sendJson(res, sale, 201);
  } else {
    res.statusCode = 404;
    res.end('Not found');
  }
}
