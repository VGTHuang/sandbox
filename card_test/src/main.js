import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import '@/assets/css/main.css'
import DynamicCard from '@/components/DynamicCard'
import CardDynamicContainer from '@/components/CardDynamicContainer'

const app = createApp(App)
app.component('DynamicCard', DynamicCard)
app.component('CardDynamicContainer', CardDynamicContainer)
app.use(router).mount('#app')
