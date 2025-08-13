import { db } from '../storage.js';
import { parseJson, sendJson } from '../utils.js';

export async function handleUsers(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === 'GET' && url.pathname === '/api/users') {
    sendJson(res, db.users);
  } else if (req.method === 'POST' && url.pathname === '/api/users') {
    const { username, role } = await parseJson(req);
    const user = { id: db.nextUserId++, username, role: role || 'user' };
    db.users.push(user);
    sendJson(res, user, 201);
  } else {
    res.statusCode = 404;
    res.end('Not found');
  }
}
