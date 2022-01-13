<template>
  <div class="demo-container"
    @click="containerClicked">
    <!-- <div class="card-button-group">
      <button @click="cardArrangement = 0">stack</button>
      <button @click="cardArrangement = 1">carousel</button>
    </div> -->
    <div class="card-container" :style="{
      transform: `rotateX(${dampRotY}deg) rotateY(${-dampRotX}deg)`
    }"
    >
      <Card v-for="(item, i) in cardContents" :key="i"
      :translate="item.translate"
      :rotate="item.rotate"
      :blur="item.blur"
      @clicked="cardClicked($event, i)"
      >
      <template v-slot:front>
        <span class="card-front-style">{{item.text}}</span>
      </template>
      </Card>
    </div>

  </div>
</template>

<script>
import Card from "@/components/Card";
import { IsMobile } from "@/js/Utils"
export default {
  components: {
    Card
  },
  data() {
    return {
      /**
       * in cardContents,
       *    translate indicates each card's translation **relative to table center**
       */
      cardContents: [
        {
          translate: [0,0,0],
          rotate: [0,0,0,0],
          blur: 0,
          text: "card1"
        },
        {
          translate: [0,0,0],
          rotate: [0,0,0,0],
          blur: 0,
          text: "card2"
        },
        {
          translate: [0,0,0],
          rotate: [0,0,0,0],
          blur: 0,
          text: "card3"
        },
        {
          translate: [0,0,0],
          rotate: [0,0,0,0],
          blur: 0,
          text: "card4"
        },
        {
          translate: [0,0,0],
          rotate: [0,0,0,0],
          blur: 0,
          text: "card5"
        },
        {
          translate: [0,0,0],
          rotate: [0,0,0,0],
          blur: 0,
          text: "card6"
        },
        {
          translate: [0,0,0],
          rotate: [0,0,0,0],
          blur: 0,
          text: "card7"
        }
      ],
      
      cardArrangement: 0,

      // mouse damp
      bufferDamp: 0.9,
      mousePosX: -1,
      mousePosY: -1,
      dampPosX: -1,
      dampPosY: -1,
      dampRotX: 0,
      dampRotY: 0,
      dampMinValue: 1e-3,
      dampRotIntensity: 0.02,
      dampInterval: undefined,

      // stack
      cardStackGap: 10,

      // carousel
      carouselWheelRadius: 180,
      carouselWheelCenter: -100,
      tempCarouselItem: 0,

      // touch
      touchBeginX: 0,
      touchBeginY: 0,
      touchPosX: 0,
      touchPosY: 0
    }
  },
  created() {
    this.cardMove__Stack()
    this.dampInterval = setInterval(this.updateMouseDamp, "50")
    window.addEventListener('mousemove', this.getMousePos)

    // calculate carousel wheel radius = card_wid/(2*atan(pi/n))
    this.carouselWheelRadius = 180/(2*Math.atan(Math.PI/this.cardContents.length))
    
    window.addEventListener("wheel", this.mouseWheel)
    if (IsMobile()) {
      window.addEventListener('touchstart',this.touchs,false)
      window.addEventListener('touchmove',this.touchs,false)
      window.addEventListener('touchend',this.touchs,false)
    }
    else {
      window.addEventListener('touchstart',this.touchs,false)
      window.addEventListener('touchmove',this.touchs,false)
      window.addEventListener('touchend',this.touchs,false)
    }
  },
  methods: {
    touchs(event) {
      let touch = event.touches[0]
      if (event.type == 'touchmove') {
        this.touchPosX = touch.clientX
        this.touchPosY = touch.clientY
      }
      else if (event.type == 'touchstart') {
        this.touchBeginX = touch.clientX
        this.touchBeginY = touch.clientY
        this.touchPosX = touch.clientX
        this.touchPosY = touch.clientY
      }
      else if (event.type == 'touchend' || event.type == 'touchcancel') {
        console.log('asd')
        let touchMoveDistX = this.touchPosX - this.touchBeginX
        let touchMoveDistY = this.touchPosY - this.touchBeginY
        if (Math.abs(touchMoveDistY) < Math.abs(touchMoveDistX) * 0.5 && Math.abs(touchMoveDistX) > 50) {
          if (this.cardArrangement !== 1) {
            return
          }
          this.mouseWheel({deltaY: touchMoveDistX})
          // if (touchMoveDistX > 0) {
          //   // swipe right
          // }
          // else {
          //   // swipe left
          // }
        }
      }
      // console.log(touch, event.type)
    },
    getMousePos(val) {
      this.mousePosX = val.clientX
      this.mousePosY = val.clientY
      if (this.dampInterval == undefined) {
        this.dampInterval = setInterval(this.updateMouseDamp, "20")
      }
    },
    updateMouseDamp() {
      if (this.dampPosX != -1 || this.dampPosY != -1) {
        this.dampPosX = this.bufferDamp * this.dampPosX + (1-this.bufferDamp) * this.mousePosX
        this.dampPosY = this.bufferDamp * this.dampPosY + (1-this.bufferDamp) * this.mousePosY
        this.dampRotX = this.mousePosX - this.dampPosX
        this.dampRotY = this.mousePosY - this.dampPosY
        this.dampRotX *= this.dampRotIntensity
        this.dampRotY *= this.dampRotIntensity
        if (Math.abs(this.dampRotX) < this.dampMinValue && Math.abs(this.dampRotY) < this.dampMinValue) {
          this.dampRotX = 0
          this.dampRotY = 0
          clearInterval(this.dampInterval)
          this.dampInterval = undefined
        }
      }
      else {
        this.dampPosX = this.mousePosX
        this.dampPosY = this.mousePosX
      }
    },
    cardMove__Stack() {
      for(let i = 0; i < this.cardContents.length; i++) {
        let tempContent = this.cardContents[i]
        
        let pos = -this.cardContents.length / 2 + i
        pos *= 10
        tempContent.rotate = [0,0,0]
        tempContent.translate = [pos, pos, -pos]
        tempContent.blur = 0
      }
    },
    cardMove__Carousel(old_index, display_index) {
      let deg = 360 / this.cardContents.length
      for(let i = 0; i < this.cardContents.length; i++) {
        let index = (i+display_index+this.cardContents.length) % this.cardContents.length
        let tempContent = this.cardContents[index]
        let tempPosX =Math.cos(2 * Math.PI * i / this.cardContents.length) * this.carouselWheelRadius
           + this.carouselWheelCenter
        let tempPosY = Math.sin(2 * Math.PI * i / this.cardContents.length) * this.carouselWheelRadius
        if (old_index != display_index) {
          tempContent.rotate = [0, tempContent.rotate[1]+deg * (old_index-display_index), 0]
        }
        else {
          tempContent.rotate = [0, i*360/this.cardContents.length, 0]
        }
        tempContent.translate = [tempPosY, 0, tempPosX]
        tempContent.blur = 0
      }
    },
    cardClicked(e, val) {
      e.stopPropagation()
      if (this.cardArrangement == 0) {
        this.tempCarouselItem = val
        this.cardArrangement = 1
        // this.cardMove__Detail(this.tempCarouselItem, val)
      }
      else if (this.cardArrangement == 1) {
        if (val != this.tempCarouselItem) {
          this.cardMove__Carousel(this.tempCarouselItem, val)
          this.tempCarouselItem = val
        }
        else {
          this.cardArrangement = 2
        }
      }
    },
    cardMove__Detail(detailedCardIndex) {
      let detailedCard = this.cardContents[detailedCardIndex]
      detailedCard.translate = [0, 0, 250]
      let cardCount = 0
      for(let i = 0; i < this.cardContents.length; i++) {
        if (detailedCardIndex != i) {
          let tempContent = this.cardContents[i]
          tempContent.translate = [(cardCount - (this.cardContents.length-1)/2) * 150+200, 0, -cardCount*200 + 100]
          tempContent.rotate = [Math.random() * 360, Math.random() * 360, Math.random() * 360]
          // tempContent.blur = 5
          cardCount += 1
        }
      }
    },
    mouseWheel(val) {
      if (this.cardArrangement == 1) {
        if (val.deltaY > 0) {
          // scroll down
          // this.tempCarouselItem += 1
          this.cardMove__Carousel(this.tempCarouselItem, this.tempCarouselItem+1)
          this.tempCarouselItem++
          this.tempCarouselItem %= this.cardContents.length
        }
        else if (val.deltaY < 0) {
          // scroll up
          // this.tempCarouselItem -= 1
          this.cardMove__Carousel(this.tempCarouselItem, this.tempCarouselItem-1)
          this.tempCarouselItem--
          this.tempCarouselItem = (this.tempCarouselItem + this.cardContents.length) % this.cardContents.length
        }
      }
    },
    containerClicked() {
      if (this.cardArrangement == 2) {
        this.cardArrangement = 1
      }
      else if (this.cardArrangement == 1) {
        this.cardArrangement = 0
      }
    }
  },
  watch: {
    cardArrangement(val) {
      switch(val) {
        case 0:
          this.cardMove__Stack()
          break
        case 1:
          this.cardMove__Carousel(this.tempCarouselItem,this.tempCarouselItem)
          break
        case 2:
          this.cardMove__Detail(this.tempCarouselItem)
          break
        // TODO
        default:
          this.cardMove__Stack()
          break
      }
    }
  }
}
</script>

<style scoped>
.demo-container {
  height: 100%;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}
.card-container {
  position: relative;
  height: 0;
  width: 0;
  perspective: 1000px;
  transform-style: preserve-3d;
  transform-origin: 0 0 0;
  display: flex;
  justify-content: center;
  align-items: center;
  /* display: flex;
  justify-content: center;
  align-items: center; */
}
.card-button-group {
  position: absolute;
  top: 100px;
  z-index: 999;
}
.card-front-style {
  font-size: 40pt;
}
</style>