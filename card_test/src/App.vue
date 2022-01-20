<template>
  <div id="nav">
    <p>
      <router-link to="/">Home</router-link> |
      <router-link to="/cards_demo_1">Cards 1</router-link> |
      <router-link to="/cards_demo_2/0">Cards 2</router-link> |
      <router-link to="/cards_demo_2/1">Cards 2 w/ bg</router-link> |
      <router-link to="/cards_demo_3">Cards 3</router-link>
    </p>
    <p>
      <router-link to="/cards_demo_perf_test_1">Performance test 1</router-link> |
      <router-link to="/cards_demo_perf_test_2">Performance test 2</router-link>
    </p>
  </div>
  <router-view id="view"/>
</template>

<style>
html {
  height: 100%;
}
body {
  margin: 0;
  height: 100%;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  padding: 0;
  margin: 0;
  height: 100%;
  background: #fff;
  overflow: hidden;
}

#nav {
  position: absolute;
  width: 100%;
  height: 50px;
  background-image: linear-gradient(#ffff 20%, #fff0);
  z-index: 1000;
}

#nav a {
  font-weight: bold;
  color: #2c3e50;
}

#nav a.router-link-exact-active {
  color: #42b983;
}
#view {
  height: 100%;
  width: 100%;
}
</style>
