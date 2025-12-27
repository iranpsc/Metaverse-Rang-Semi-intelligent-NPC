# Nginx WebSocket Configuration Fix

## Problem
WebSocket connections are failing because nginx is stripping the `Upgrade` and `Connection` headers required for WebSocket handshakes.

## Solution
Add the following configuration to your nginx server block for WebSocket endpoints:

```nginx
location /ws/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    
    # WebSocket upgrade headers - CRITICAL
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    
    # Standard proxy headers
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # WebSocket timeouts
    proxy_read_timeout 86400;
    proxy_send_timeout 86400;
}
```

## Key Points

1. **`proxy_set_header Upgrade $http_upgrade;`** - This preserves the `Upgrade: websocket` header
2. **`proxy_set_header Connection "upgrade";`** - This sets `Connection: upgrade` (not "close")
3. **`proxy_http_version 1.1;`** - Required for WebSocket upgrades
4. **Timeouts** - WebSocket connections can be long-lived, so increase timeouts

## Testing

After updating nginx configuration:
1. Reload nginx: `sudo nginx -s reload` or `sudo systemctl reload nginx`
2. Test the WebSocket connection from the browser
3. Check server logs to verify WebSocket connections are being accepted

## Alternative: Direct Connection

If you want to bypass nginx entirely for WebSocket connections:
1. Ensure port 8000 is accessible from the browser
2. Update the frontend to connect directly to `ws://your-domain:8000/ws/rag-chat/`
3. Note: This may require firewall/security group changes

