// server/static/dashboard.js
document.addEventListener('DOMContentLoaded', function(){
  const socket = io();
  socket.on('connect', () => {
    console.log('socket connected');
  });
  socket.on('new_prediction', (msg) => {
    const div = document.getElementById('predictions');
    const p = document.createElement('div');
    p.className = 'pred';
    p.innerText = `${new Date().toLocaleTimeString()} — amount: ${msg.tx.amount} → ${msg.pred}`;
    div.prepend(p);
  });
});

function removeClient(){
  const admin_pwd = document.getElementById('admin_pwd').value;
  const cid = document.getElementById('remove_cid').value;
  fetch('/api/admin/remove_client', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({admin_password: admin_pwd, client_id: cid})
  }).then(r => r.json()).then(j => {
    document.getElementById('admin_status').innerText = JSON.stringify(j);
    setTimeout(()=>location.reload(), 900);
  });
}
