document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if(targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if(targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Sticky navigation highlight
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-links a');
    const scrollTopBtn = document.querySelector('.scroll-top');
    
    window.addEventListener('scroll', () => {
        let current = '';
        
        // Show/hide scroll to top button
        if (window.pageYOffset > 300) {
            scrollTopBtn.classList.add('show');
        } else {
            scrollTopBtn.classList.remove('show');
        }
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if(pageYOffset >= (sectionTop - 100)) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if(link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
    
    // Scroll to top button functionality
    scrollTopBtn.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    // Add reveal animations when scrolling
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };
    
    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('show');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.feature-box').forEach(item => {
        observer.observe(item);
    });
    
    // Section visibility animation
    const sectionObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        root: null,
        rootMargin: '-100px',
        threshold: 0.1
    });
    
    document.querySelectorAll('section').forEach(section => {
        sectionObserver.observe(section);
    });
    
    // Image modal functionality
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const modalClose = document.querySelector('.modal-close');
    const resultImages = document.querySelectorAll('.result-img');
    
    resultImages.forEach(img => {
        img.addEventListener('click', function() {
            modal.classList.add('show');
            modalImg.src = this.getAttribute('data-full');
        });
    });
    
    modalClose.addEventListener('click', function() {
        modal.classList.remove('show');
    });
    
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.classList.remove('show');
        }
    });
    
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.classList.contains('show')) {
            modal.classList.remove('show');
        }
    });

    // Animation for the conclusion contributions
    const contributionList = document.querySelector('.contribution-list');
    if (contributionList) {
        const contributionItems = document.querySelectorAll('.contribution-item');
        
        const animateContributions = () => {
            contributionItems.forEach((item, index) => {
                setTimeout(() => {
                    item.classList.add('visible');
                }, index * 400);
            });
        };
        
        const contributionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateContributions();
                    contributionObserver.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.5
        });
        
        contributionObserver.observe(contributionList);
    }

    // Interactive demo functionality
    const imageOptions = document.querySelectorAll('.image-option');
    const queryDogFlower = document.getElementById('query-dog-flower');
    const queryLampMug = document.getElementById('query-lamp-mug');
    const attentionVisualization = document.querySelector('.attention-visualization');
    const clipAttentionImg = document.getElementById('clip-attention');
    const diffclipAttentionImg = document.getElementById('diffclip-attention');
    
    // Step 1: Image selection
    imageOptions.forEach(option => {
        option.addEventListener('click', function() {
            // Remove selection from all options
            imageOptions.forEach(opt => opt.classList.remove('selected'));
            
            // Add selection to clicked option
            this.classList.add('selected');
            
            // Hide all query selections
            queryDogFlower.style.display = 'none';
            queryLampMug.style.display = 'none';
            
            // Hide visualization
            attentionVisualization.style.display = 'none';
            
            // Show relevant query options based on selected image
            const selectedImage = this.getAttribute('data-image');
            if (selectedImage.includes('a-cute-running-after-a-ball')) {
                queryDogFlower.style.display = 'block';
            } else if (selectedImage.includes('mystery-case-files')) {
                queryLampMug.style.display = 'block';
            }
        });
    });
    
    // Step 2: Query selection
    const queryButtons = document.querySelectorAll('.query-btn');
    queryButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            // Remove selection from all query buttons in the same group
            this.parentNode.querySelectorAll('.query-btn').forEach(b => {
                b.classList.remove('selected');
            });
            
            // Add selection to clicked button
            this.classList.add('selected');
            
            // Get the selected query and base image
            const query = this.getAttribute('data-query');
            const baseImage = this.getAttribute('data-base-image');
            
            // Update visualization images
            clipAttentionImg.src = `images/individual_images/${baseImage}_${query}_CLIP.png`;
            diffclipAttentionImg.src = `images/individual_images/${baseImage}_${query}_DiffCLIP.png`;
            
            // Show the visualization
            attentionVisualization.style.display = 'block';
            
            // Scroll to visualization
            setTimeout(() => {
                attentionVisualization.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        });
    });
});