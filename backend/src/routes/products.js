import { db } from '../storage.js';
import { parseJson, sendJson } from '../utils.js';

export async function handleProducts(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const idMatch = url.pathname.match(/^\/api\/products\/(\d+)/);

  if (req.method === 'GET' && url.pathname === '/api/products') {
    sendJson(res, db.products);
  } else if (req.method === 'POST' && url.pathname === '/api/products') {
    const { name, stock } = await parseJson(req);
    const product = { id: db.nextProductId++, name, stock: Number(stock) || 0 };
    db.products.push(product);
    sendJson(res, product, 201);
  } else if (req.method === 'PUT' && idMatch) {
    const id = Number(idMatch[1]);
    const product = db.products.find(p => p.id === id);
    if (!product) return sendJson(res, { error: 'Product not found' }, 404);
    const { name, stock } = await parseJson(req);
    if (name !== undefined) product.name = name;
    if (stock !== undefined) product.stock = Number(stock);
    sendJson(res, product);
  } else if (req.method === 'DELETE' && idMatch) {
    const id = Number(idMatch[1]);
    const index = db.products.findIndex(p => p.id === id);
    if (index === -1) return sendJson(res, { error: 'Product not found' }, 404);
    const [removed] = db.products.splice(index, 1);
    sendJson(res, removed);
  } else {
    res.statusCode = 404;
    res.end('Not found');
  }
}
