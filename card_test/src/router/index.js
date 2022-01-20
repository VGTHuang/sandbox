import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/cards_demo_1',
    name: 'CardsDemo1',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () => import(/* webpackChunkName: "about" */ '../views/CardsDemo1.vue')
  },
  {
    path: '/cards_demo_2/:bg',
    name: 'CardsDemo2',
    component: () => import('../views/CardsDemo2.vue')
  },
  {
    path: '/cards_demo_3',
    name: 'CardsDemo3',
    component: () => import('../views/CardsDemo3.vue')
  },
  {
    path: '/cards_demo_perf_test_1',
    name: 'CardsDemoPerfTest1',
    component: () => import('../views/CardsDemoPerfTest1.vue')
  },
  {
    path: '/cards_demo_perf_test_2',
    name: 'CardsDemoPerfTest2',
    component: () => import('../views/CardsDemoPerfTest2.vue')
  },
  // {
  //   path: '/cards_demo_threejs',
  //   name: 'CardsDemoThree',
  //   component: () => import('../views/CardsDemoThree.vue')
  // }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
