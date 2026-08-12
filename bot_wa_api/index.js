const { makeWASocket, useMultiFileAuthState, DisconnectReason } = require('@whiskeysockets/baileys');
const pino = require('pino');
const qrcode = require('qrcode-terminal');
const express = require('express');
const multer = require('multer');

const app = express();
const port = 1016;
const upload = multer({ storage: multer.memoryStorage() });

let sock;

async function connectToWhatsApp () {
    const { state, saveCreds } = await useMultiFileAuthState('sesi_bot_baileys');
    
    sock = makeWASocket({
        auth: state,
        printQRInTerminal: false, // Kita print QR manual agar rapi
        logger: pino({ level: 'silent' }) // Matikan log bawaan yang terlalu ramai
    });

    sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;
        
        if (qr) {
            console.log('\n=========================================');
            console.log('SCAN QR CODE INI DENGAN WHATSAPP ANDA:');
            qrcode.generate(qr, { small: true });
            console.log('=========================================\n');
        }

        if (connection === 'close') {
            const shouldReconnect = (lastDisconnect.error)?.output?.statusCode !== DisconnectReason.loggedOut;
            console.log('❌ Koneksi terputus. Reconnecting:', shouldReconnect);
            if (shouldReconnect) {
                connectToWhatsApp();
            }
        } else if (connection === 'open') {
            console.log('✅ Bot WhatsApp (Metode Baileys) sudah siap dan terkoneksi!');
        }
    });

    sock.ev.on('creds.update', saveCreds);
}

// Endpoint untuk menerima gambar & caption dari Python
app.post('/send', upload.single('image'), async (req, res) => {
    try {
        const { phone, caption } = req.body;
        
        if (!phone) {
            return res.status(400).json({ error: 'Nomor HP wajib diisi' });
        }
        if (!sock) {
            return res.status(500).json({ error: 'Bot WA belum siap' });
        }

        // Format nomor Baileys: 628123456789@s.whatsapp.net
        let formattedPhone = phone.replace(/\D/g, '');
        if (formattedPhone.startsWith('0')) {
            formattedPhone = '62' + formattedPhone.substring(1);
        }
        if (!formattedPhone.endsWith('@s.whatsapp.net')) {
            formattedPhone += '@s.whatsapp.net';
        }

        let sentMsg;
        if (req.file) {
            sentMsg = await sock.sendMessage(formattedPhone, { 
                image: req.file.buffer, 
                caption: caption || '' 
            });
        } else {
            sentMsg = await sock.sendMessage(formattedPhone, { 
                text: caption || '' 
            });
        }

        console.log(`[INFO] Pesan berhasil dikirim ke ${formattedPhone}`);
        res.json({ status: 'success', message: 'Pesan berhasil dikirim' });

    } catch (error) {
        console.error('[ERROR] Gagal mengirim pesan:', error);
        res.status(500).json({ status: 'error', message: error.toString() });
    }
});

app.listen(port, '0.0.0.0', () => {
    console.log(`🚀 Server API Bot WA (Baileys) berjalan di port ${port}`);
    connectToWhatsApp();
});
