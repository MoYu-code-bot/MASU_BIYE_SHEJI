<template>
  <div class="customer-home">
    <!-- 公告滚动条 -->
    <div class="announcement-bar" v-if="announcements.length > 0">
      <div class="announcement-icon">📢</div>
      <div class="announcement-scroll">
        <div class="announcement-track" :style="announcementTrackStyle">
          <span
            v-for="(item, i) in announcements"
            :key="i"
            class="announcement-item"
          >
            【{{ item.title }}】{{ item.content }}
          </span>
        </div>
      </div>
    </div>

    <!-- 轮播图 -->
    <el-carousel height="560px" class="banner-carousel" v-if="banners.length > 0">
      <el-carousel-item v-for="banner in banners" :key="banner.id">
        <img :src="banner.image" :alt="banner.title" class="banner-image" />
      </el-carousel-item>
    </el-carousel>

    <!-- 门店列表 -->
    <h2 class="section-title">附近门店</h2>
    <el-row :gutter="20">
      <el-col :span="8" v-for="store in stores" :key="store.id">
        <el-card shadow="hover" class="store-card" @click="goToStore(store.id)">
          <div class="store-cover">
            <img :src="store.coverImage || defaultCover" :alt="store.name" />
          </div>
          <div class="store-info">
            <h3 class="store-name">{{ store.name }}</h3>
            <p class="store-address">{{ store.address }}</p>
            <p class="store-hours">营业时间: {{ store.businessHours }}</p>
            <p class="store-price">人均: <span class="price-value">¥{{ store.avgPrice }}</span></p>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-empty v-if="stores.length === 0 && !loading" description="暂无门店" />
  </div>
</template>

<script setup>
import { ref, onMounted, computed, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getBanners, getStores, getAnnouncements } from '@/api'

const router = useRouter()
const banners = ref([])
const stores = ref([])
const announcements = ref([])
const loading = ref(false)
const announcementOffset = ref(0)

const defaultCover = 'https://via.placeholder.com/400x200?text=火锅'

const announcementTrackStyle = computed(() => ({
  transform: `translateX(-${announcementOffset.value}px)`
}))

let announcementTimer = null

const loadBanners = async () => {
  try {
    const res = await getBanners({ page: 1, pageSize: 10 })
    banners.value = (res.data?.records || res.data?.list || res.data || []).filter(b => b.status === 1)
  } catch (e) {
    // ignore
  }
}

const loadAnnouncements = async () => {
  try {
    const res = await getAnnouncements()
    announcements.value = res.data || []
    if (announcements.value.length > 0) {
      startAnnouncementScroll()
    }
  } catch (e) {
    // ignore
  }
}

const startAnnouncementScroll = () => {
  const speed = 1 // 滚动速度 px/帧
  const animate = () => {
    announcementOffset.value += speed
    // 重置实现无缝循环
    const barEl = document.querySelector('.announcement-scroll')
    if (barEl) {
      const trackEl = barEl.querySelector('.announcement-track')
      if (trackEl && announcementOffset.value >= trackEl.scrollWidth / 2) {
        announcementOffset.value = 0
      }
    }
    announcementTimer = requestAnimationFrame(animate)
  }
  announcementTimer = requestAnimationFrame(animate)
}

const loadStores = async () => {
  loading.value = true
  try {
    const res = await getStores({ page: 1, pageSize: 20 })
    stores.value = (res.data?.records || res.data?.list || res.data || []).filter(s => s.status === 1)
  } catch (e) {
    ElMessage.error('加载门店列表失败')
  } finally {
    loading.value = false
  }
}

const goToStore = (storeId) => {
  router.push(`/c/stores/${storeId}`)
}

onMounted(() => {
  loadAnnouncements()
  loadBanners()
  loadStores()
})

onUnmounted(() => {
  if (announcementTimer) {
    cancelAnimationFrame(announcementTimer)
  }
})
</script>

<style scoped>
.customer-home {
  padding-bottom: 20px;
}

/* 公告滚动条 */
.announcement-bar {
  display: flex;
  align-items: center;
  background: linear-gradient(135deg, #fff8f0, #ffe8d6);
  border: 1px solid #ffb366;
  border-radius: 8px;
  padding: 10px 16px;
  margin-bottom: 16px;
  overflow: hidden;
}

.announcement-icon {
  font-size: 20px;
  margin-right: 12px;
  flex-shrink: 0;
  animation: iconPulse 1.5s ease-in-out infinite;
}

@keyframes iconPulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.15); }
}

.announcement-scroll {
  flex: 1;
  overflow: hidden;
  position: relative;
}

.announcement-track {
  display: inline-block;
  white-space: nowrap;
  will-change: transform;
}

.announcement-item {
  display: inline-block;
  font-size: 14px;
  color: #8b4513;
  line-height: 1.6;
  padding-right: 80px;
}

.banner-carousel {
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 30px;
}

.banner-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.section-title {
  font-size: 22px;
  margin-bottom: 20px;
  color: #333;
}

.store-card {
  margin-bottom: 20px;
  cursor: pointer;
  transition: transform 0.2s;
}

.store-card:hover {
  transform: translateY(-4px);
}

.store-card :deep(.el-card__body) {
  padding: 0;
}

.store-cover {
  width: 100%;
  height: 200px;
  overflow: hidden;
}

.store-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.store-info {
  padding: 16px;
}

.store-name {
  font-size: 18px;
  margin: 0 0 8px 0;
  color: #333;
}

.store-address,
.store-hours {
  font-size: 13px;
  color: #999;
  margin: 4px 0;
}

.store-price {
  font-size: 14px;
  color: #666;
  margin: 8px 0 0 0;
}

.price-value {
  color: #ff6b35;
  font-weight: bold;
  font-size: 16px;
}
</style>
