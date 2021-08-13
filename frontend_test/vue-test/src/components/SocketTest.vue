<template>
    <div>
        <button @click="connectToSocket">connect to socket</button>
        <button @click="disconnectSocket">disconnect socket</button>
        <button @click="sendMessage">send message</button>
    </div>
</template>

<script>
export default {
    data() {
        return {
            backend: 'ws://127.0.0.1:5000/api',
            websocket: undefined
        }
    },
    methods: {
        connectToSocket() {
            let self = this
            console.log('connect')
            this.websocket = new WebSocket(this.backend)
            this.websocket.onopen = function() {
                console.log('open', self.websocket)
                self.websocket.send('New participant joined');
            };
            this.websocket.onmessage = function(e) {
                console.log('message received', e)
            };
        },
        disconnectSocket() {
            if(this.websocket) {
                this.websocket.close()
            }
        },
        sendMessage() {
            this.websocket.send(JSON.stringify({
                's': 'asdasdasd'
            }));
        }
    }

}
</script>