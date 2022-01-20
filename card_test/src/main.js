import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import { Picture } from '@element-plus/icons-vue'
import '@/assets/css/main.css'
import DynamicCard from '@/components/DynamicCard'
import CardDynamicContainer from '@/components/CardDynamicContainer'
import InnerDynamicContainer from '@/components/InnerDynamicContainer'

const app = createApp(App)
app.component('DynamicCard', DynamicCard)
app.component('CardDynamicContainer', CardDynamicContainer)
app.component('InnerDynamicContainer', InnerDynamicContainer)
const components = [Picture]
for (const iconName of components) {
  app.component(iconName.name, iconName)
}
app.use(router).use(ElementPlus).mount('#app')
