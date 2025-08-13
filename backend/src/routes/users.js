import { Router } from 'express';
import { db } from '../storage.js';

const router = Router();

router.get('/', (req, res) => {
  res.json(db.users);
});

router.post('/', (req, res) => {
  const { username, role } = req.body;
  const user = { id: db.nextUserId++, username, role: role || 'user' };
  db.users.push(user);
  res.status(201).json(user);
});

export default router;
