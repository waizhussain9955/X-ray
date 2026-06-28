import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { DEFAULT_PORT } from '../shared/const';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || DEFAULT_PORT;

// Serve static assets from the build directory
const publicPath = path.resolve(__dirname, 'public');
app.use(express.static(publicPath));

// Fallback route for SPA
app.get('*', (_req, res) => {
  res.sendFile(path.resolve(publicPath, 'index.html'));
});

app.listen(port, () => {
  console.log(`Express server running on http://localhost:${port}`);
  console.log(`Serving static assets from: ${publicPath}`);
});
