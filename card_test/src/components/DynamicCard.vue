<template>
  <div class="card__basic-style"
  ref="card"
  :class="{transition__basic: doTransition, 'is-front': isFront}"
  :style="{
    transform: cardTransform
  }"
  @click="$emit('clicked', $event)"
  >
    <div class="card__front">
      <button @click="bringToFront">bring to front</button>
      <button @click="bringToBack">bring to back</button>
      <slot name="front">card</slot>
    </div>
    <div class="card__back">
      <slot name="back">card</slot>
    </div>
    <slot name="icing"></slot>
  </div>
</template>

<script>
export default {
  name: "DynamicCard",
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
    },
    doTransition: {
      type: Boolean,
      default: true,
    }
  },
  data() {
    return {
      isFront: false,
      card: undefined
    }
  },
  mounted() {
    this.card = this.$refs['card']
  },
  computed: {
    cardTransform() {
      if (this.isFront) {
        return `translateZ(${200}pt)`
      }
      else {
        return `translateX(${this.translate[0]}pt)
          translateY(${this.translate[1]}pt)
          translateZ(${this.translate[2]}pt)
          rotateX(${this.rotate[0]}deg)
          rotateY(${this.rotate[1]}deg)
          rotateZ(${this.rotate[2]}deg)`
      }
    }
  },
  methods: {
    bringToFront() {
      this.isFront = true
      this.$emit('front', this.card)
      // this.translate = [0,0,0]
      // this.rotate = [0,0,0]
    },
    bringToBack() {
      this.isFront = false
      this.$emit('back', this.card)
      // this.translate = [0,0,0]
      // this.rotate = [0,0,0]
    }
  }
}
</script>

<style scoped>
.card__back {
  background-image: url('https://opengameart.org/sites/default/files/card%20back%20black.png');
  background-size: 180pt 252pt;
}
</style>