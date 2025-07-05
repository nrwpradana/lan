from newspaper import Article


def scrape(url):
    """Scrap article from url

    Args:
        url (`string`): news url

    Returns:
        article (`string`): news content
    """

    article = Article(url, language="id")
    article.download()
    article.parse()

    if not article.text:
        print("Can't Scrap this article link")
        return None

    return article
