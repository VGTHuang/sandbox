import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import '@/assets/css/main.css'
import DynamicCard from '@/components/DynamicCard'
import CardDynamicContainer from '@/components/CardDynamicContainer'
import InnerDynamicContainer from '@/components/InnerDynamicContainer'

const app = createApp(App)
app.component('DynamicCard', DynamicCard)
app.component('CardDynamicContainer', CardDynamicContainer)
app.component('InnerDynamicContainer', InnerDynamicContainer)
app.use(router).mount('#app')
