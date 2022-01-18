<template>
  <div class="inner-container transition__basic" :style="{
    transform: `translateX(${translate[0]}pt) translateY(${translate[1]}pt) translateZ(${translate[2]}pt)
    rotateX(${rotate[0]}deg) rotateY(${rotate[1]}deg) rotateZ(${rotate[2]}deg)`
    }">
    <slot></slot>
  </div>
</template>

<script>
export default {
  name: "InnerDynamicContainer",
  props: {
    translate: {
      type: Array,
      default: () => [0,0,0],
    },
    rotate: {
      type: Array,
      default: () => [0,0,0],
    }
  },
}
</script>

<style scoped>
.inner-container {
  display: flex;
  justify-content: center;
  align-items: center;
  transform: translateX(0pt) translateY(0pt) translateZ(0pt) rotateX(0deg) rotateY(0deg) rotateZ(0deg);
  transform-style: preserve-3d;
}
</style>