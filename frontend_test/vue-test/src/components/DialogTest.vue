<template>
  <div class="move-item"
   ref="move-item"
    :style="{width: sizeX+'px', height: sizeY+'px', top: tempY+'px', left: tempX+'px'}"
  >
    <div class="drag-bar"
      @dragover.prevent
      draggable="true"
      @dragstart="onDragStart"
      @drag="onDrag"
    ></div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      sizeX: 300,
      sizeY: 300,
      tempX: 0,
      tempY: 0,
      mouseDownX: 0,
      mouseDownY: 0,
      mouseDownRectX: 0,
      mouseDownRectY: 0,
      ghostImage: undefined
    }
  },
  created() {
    window.addEventListener("resize", this.checkOutOfBounds)
    this.ghostImage = document.createElement('img');
    this.ghostImage.src = ''
    this.ghostImage.alt = ''
  },
  methods: {
    checkOutOfBounds() {
      if (this.tempX > window.innerWidth - this.sizeX) {
        this.tempX = window.innerWidth - this.sizeX
      }
      if (this.tempY > window.innerHeight - this.sizeY) {
        this.tempY = window.innerHeight - this.sizeY
      }
    },
    onDragStart(e) {
      this.mouseDownX = e.clientX
      this.mouseDownY = e.clientY
      let rect = this.$refs["move-item"].getClientRects()[0]
      this.mouseDownRectX = rect.left
      this.mouseDownRectY = rect.top
      e.dataTransfer.setDragImage(this.ghostImage, 0, 0)
    },
    onDrag(e) {
      if (e.clientX <= 0) {
        return
      }
      this.tempX = e.clientX + this.mouseDownRectX - this.mouseDownX
      this.tempY = e.clientY + this.mouseDownRectY - this.mouseDownY
      if (this.tempX < 0) {
        this.tempX = 0
      } else if (this.tempX > window.innerWidth - this.sizeX) {
        this.tempX = window.innerWidth - this.sizeX
      }
      if (this.tempY < 0) {
        this.tempY = 0
      } else if (this.tempY > window.innerHeight - this.sizeY) {
        this.tempY = window.innerHeight - this.sizeY
      }
    }
  },
  beforeUnmount() {
    window.removeEventListener("resize", this.checkOutOfBounds)
  }
}
</script>

<style scoped>

.move-item {
  background-color: #d0d0d0;
  position: fixed;
}
.drag-bar {
  background-color: #a0a0a0;
  height: 50px;
  width: 100%;
}

</style>