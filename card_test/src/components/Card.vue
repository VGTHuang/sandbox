<template>
  <div class="card"
  :style="{
    transform: `translateX(${translate[0]}pt) translateY(${translate[1]}pt) translateZ(${translate[2]}pt) rotateX(${rotate[0]}deg) rotateY(${rotate[1]}deg) rotateZ(${rotate[2]}deg)`,
    
  }"
  @click="$emit('clicked', $event)"
  >
    <div class="card_front">
      <slot name="front">card</slot>
    </div>
    <div class="card_back">
      <slot name="back">card</slot>
    </div>
  </div>
</template>

<script>
export default {
  name: "Card",
  props: {
    translate: {
      type: Array,
      default: () => [0,0,0],
    },
    rotate: {
      type: Array,
      default: () => [0,0,0],
    },
    blur: {
      type: Number,
      default: 0,
    }
  }
}
</script>

<style scoped>
.card {
  position: absolute;
  border: 1px solid red;
  border-radius: 15px;
  height: 252pt; /* 7 * 36 */
  width: 180pt; /* 5 * 36 */
  background: #ccca;
  transform-origin: 90pt 126pt 0;
  transform-style: preserve-3d;
  transition: cubic-bezier(0.165, 0.84, 0.44, 1) 250ms;
  /* overflow: hidden; */
}
.card_front, .card_back {
  height: 252pt; /* 7 * 36 */
  width: 180pt; /* 5 * 36 */
  position: absolute;
  backface-visibility: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.card_back {
  background-image: url('https://i.pinimg.com/originals/69/d2/31/69d2315be22e9d4e9cb53d3c4f33fc3d.jpg');
  background-size: 180pt 252pt;
  transform: rotateY(180deg);
}
</style>