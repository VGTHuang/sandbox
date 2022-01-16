<template>
  <card-dynamic-container @containerClicked="containerClicked">
      <dynamic-card v-for="(item, i) in cardContents" :key="i"
      :translate="item.translate"
      :rotate="item.rotate"
      :blur="item.blur"
      @clicked="cardClicked($event, i)"
      >
        <template v-slot:front>
          <span class="card-front-style">{{item.text}}</span>
        </template>
      </dynamic-card>
    </card-dynamic-container>
</template>

<script>
export default {
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

      // stack
      cardStackGap: 10,

      // carousel
      carouselWheelRadius: 180,
      carouselWheelCenter: -100,
      tempCarouselItem: 0,
    }
  },
  created() {
    this.cardMove__Stack()
    // calculate carousel wheel radius = card_wid/(2*atan(pi/n))
    this.carouselWheelRadius = 180/(2*Math.atan(Math.PI/this.cardContents.length))
  },
  methods: {
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
.card-front-style {
  font-size: 40pt;
}
</style>