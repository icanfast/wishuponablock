API="https://dev.wishuponablock.com/api"   # or your workers.dev URL
COOKIES="/tmp/wub-auth.cookies"
EMAIL="test+$(date +%s)@example.com"
PASS="YourStrongPass123!"

curl -i -c "$COOKIES" \
  -H "content-type: application/json" \
  -X POST "$API/auth/email/signup" \
  --data "{\"email\":\"$EMAIL\",\"password\":\"$PASS\",\"username\":\"tester\"}"
