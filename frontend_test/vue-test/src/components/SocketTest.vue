<template>
    <div>
        <button @click="connectToSocket">connect to socket</button>
    </div>
</template>

<script>
export default {
    data() {
        return {
            backend: 'asdf'
        }
    },
    methods: {
        connectToSocket() {
            console.log('connect')
        }
    }

}
</script>