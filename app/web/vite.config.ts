import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: '0.0.0.0',
		proxy: {
			'/api': 'http://127.0.0.1:8002',
			'/events': { target: 'http://127.0.0.1:8002', changeOrigin: true, ws: false },
		},
	},
});
