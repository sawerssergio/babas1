import http from 'http';
import { handleProducts } from './routes/products.js';
import { handleSales } from './routes/sales.js';
import { handleUsers } from './routes/users.js';

const server = http.createServer((req, res) => {
  if (req.url.startsWith('/api/products')) {
    handleProducts(req, res);
  } else if (req.url.startsWith('/api/sales')) {
    handleSales(req, res);
  } else if (req.url.startsWith('/api/users')) {
    handleUsers(req, res);
  } else {
    res.statusCode = 404;
    res.end('Not found');
  }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
